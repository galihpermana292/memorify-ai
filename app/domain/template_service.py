# app/domain/template_service.py
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import List, Dict, Tuple
import traceback, base64, requests, os
import numpy as np, cv2
from PIL import Image

REQUEST_TIMEOUT = 15
WORKERS_IO = 8
WORKERS_CROP = 4          # keep modest; YOLO is heavy
WORKERS_COMPOSITE = 3     # how many pages composite in parallel
MAX_PAGES_IN_FLIGHT = 2   # cap memory: at most N pages in memory

SAVE_FORMAT = "jpg"
JPEG_QUALITY = 88

class TemplateService:
    def __init__(self, yolo):
        self.yolo = yolo
        os.makedirs("outputs", exist_ok=True)

    # ---------- IO helpers ----------
    def _load_image_bytes(self, src: str) -> bytes:
        if src.startswith(("http://", "https://")):
            r = requests.get(src, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.content
        if os.path.isfile(src):
            with open(src, "rb") as f:
                return f.read()
        if src.startswith("data:image"):
            _, encoded = src.split(",", 1)
            return base64.b64decode(encoded + "===")
        return base64.b64decode(src + "===")

    def _decode_cv(self, b: bytes, imread_flag=cv2.IMREAD_COLOR, max_side=1280):
        arr = np.frombuffer(b, np.uint8)
        img = cv2.imdecode(arr, imread_flag)
        if img is None:
            raise ValueError("Failed to decode image")
        h, w = img.shape[:2]
        m = max(h, w)
        if m > max_side:
            scale = max_side / m
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        return img

    def _load_many_bytes(self, sources: List[str]) -> List[bytes]:
        out = [None] * len(sources)
        with ThreadPoolExecutor(max_workers=WORKERS_IO) as ex:
            futs = {ex.submit(self._load_image_bytes, src): i for i, src in enumerate(sources)}
            for fut in as_completed(futs):
                i = futs[fut]
                out[i] = fut.result()
        return out

    # ---------- crop/composite ----------
    def _crop_one(self, img_bytes: bytes, slot) -> Image.Image:
        img_cv = self._decode_cv(img_bytes, cv2.IMREAD_COLOR)
        cropped_cv = self.yolo.crop_person(img_cv, slot.w, slot.h)  # uses no_grad, classes=[0], imgsz
        pil_img = Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))
        del img_cv, cropped_cv
        return pil_img

    def _crop_photos_for_slots_parallel(self, images_bytes: List[bytes], slots) -> List[Image.Image]:
        if len(images_bytes) < len(slots):
            raise ValueError(f"Not enough images: need {len(slots)}, got {len(images_bytes)}")
        tasks = list(zip(images_bytes[:len(slots)], slots))
        out: List[Image.Image] = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=WORKERS_CROP) as ex:
            futs = {ex.submit(self._crop_one, img_b, slot): idx for idx, (img_b, slot) in enumerate(tasks)}
            for fut in as_completed(futs):
                out[futs[fut]] = fut.result()
        return out

    def _composite_one_page(
        self, index_str: str, frame_bytes: bytes,
        cropped_for_page: List[Image.Image], slots, svg_groups, out_id: str
    ) -> Tuple[int, str]:
        from app.infrastructure.cv.image_process import insert_photos_with_svg_mask_new
        from app.infrastructure.cloudinary.upload_file import upload_pil_image

        frame_cv = self._decode_cv(frame_bytes, cv2.IMREAD_COLOR)
        final_pil = insert_photos_with_svg_mask_new(
            frame_cv, cropped_for_page, [s.model_dump() for s in slots], svg_groups
        )
        del frame_cv

        if final_pil is None:
            # Free cropped images
            for im in cropped_for_page:
                if im: im.close()
            raise ValueError("Compositing failed")

        public_id = f"{out_id}_page_{index_str}"
        url = upload_pil_image(final_pil, public_id=public_id, folder="images-scrapbook",
                               fmt=SAVE_FORMAT, quality=JPEG_QUALITY)

        # free ASAP
        final_pil.close()
        for im in cropped_for_page:
            if im: im.close()

        return (int(index_str), url)

    # ---------- streaming pipeline ----------
    def _process_one_page(self, k: str, template_data):
        page = template_data.frame_images[k]
        slots = page.photo_slots
        n = len(slots)

        # Take first N uploads for this page only (your upstream can map per page instead)
        page_sources = template_data.uploaded_images[:n]
        template_data.uploaded_images = template_data.uploaded_images[n:]

        # Load + crop for this page
        upload_bytes = self._load_many_bytes(page_sources)
        cropped_for_page = self._crop_photos_for_slots_parallel(upload_bytes, slots)
        # Free raw bytes
        del upload_bytes

        # Load the frame for this page
        frame_bytes = self._load_image_bytes(page.frame_image_path)

        # Composite and upload
        return self._composite_one_page(
            k, frame_bytes, cropped_for_page, slots, page.svg_paths, template_data.id
        )

    def process_template(self, template_data):
        results = {"run_id": template_data.id, "pages": [], "errors": []}
        page_keys_sorted = sorted(template_data.frame_images.keys(), key=lambda k: int(k))

        in_flight = []
        with ThreadPoolExecutor(max_workers=WORKERS_COMPOSITE) as ex:
            for k in page_keys_sorted:
                # backpressure to cap RAM
                while len(in_flight) >= MAX_PAGES_IN_FLIGHT:
                    done, not_done = wait(in_flight, return_when=FIRST_COMPLETED)
                    in_flight = list(not_done)
                    for fut in done:
                        try:
                            page_idx, url = fut.result()
                            results["pages"].append({"index": page_idx, "url": url})
                        except Exception as e:
                            results["errors"].append({"index": -1, "error": f"{type(e).__name__}: {e}"})

                in_flight.append(ex.submit(self._process_one_page, k, template_data))

            # drain remaining
            for fut in as_completed(in_flight):
                try:
                    page_idx, url = fut.result()
                    results["pages"].append({"index": page_idx, "url": url})
                except Exception as e:
                    results["errors"].append({"index": -1, "error": f"{type(e).__name__}: {e}"})

        results["pages"].sort(key=lambda x: x["index"])
        return results
