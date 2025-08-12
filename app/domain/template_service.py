# app/domain/template_service.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import traceback, base64, requests, os, io
import numpy as np, cv2
from PIL import Image

from app.delivery.schemas.body import TemplateData 

from app.infrastructure.cv import image_process
from app.infrastructure.cloudinary.upload_file import upload_pil_image
from app.domain.yolo_processor import YOLOProcessor

# --- KONFIGURASI OPTIMASI ---
REQUEST_TIMEOUT = 15
WORKERS_IO = 8
WORKERS_CROP = 6
WORKERS_COMPOSITE = 3

SAVE_FORMAT = "jpeg"
JPEG_QUALITY = 88

class TemplateService:
    def __init__(self, yolo: YOLOProcessor):
        self.yolo_processor = yolo
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

    def _decode_cv(self, b: bytes, imread_flag=cv2.IMREAD_COLOR, max_side=1024):
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
        img_cv = self._decode_cv(img_bytes) 
        slot_dict = slot.model_dump()
        
        cropped_cv = self.yolo_processor.crop_person(
            img_cv, int(slot_dict['w']), int(slot_dict['h'])
        )
        
        pil_img = Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))
        del img_cv, cropped_cv
        return pil_img

    def _crop_photos_for_slots_parallel(self, images_bytes: List[bytes], slots) -> List[Image.Image]:
        num_slots = len(slots)
        num_images = len(images_bytes)
        
        images_to_process = list(images_bytes) 

        if num_images < num_slots:
            print(f"⚠️ Jumlah gambar ({num_images}) lebih sedikit dari slot ({num_slots}). Gambar akan diduplikasi.")
            for i in range(num_slots - num_images):
                images_to_process.append(images_bytes[i % num_images])
        
        tasks = list(zip(images_to_process[:num_slots], slots))
        
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
        frame_pil = Image.open(io.BytesIO(frame_bytes)).convert("RGBA")
        slots_as_dicts = [s.model_dump() for s in slots]
        
        final_pil = image_process.insert_photos_with_svg_mask(
            frame_pil, cropped_for_page, slots_as_dicts, svg_groups
        )
        del frame_pil

        if final_pil is None:
            for im in cropped_for_page:
                if im: im.close()
            raise ValueError("Compositing failed")
        
        # Upload ke Cloudinary
        public_id = f"{out_id}_page_{index_str}"
        url = upload_pil_image(final_pil.convert('RGB'), public_id=public_id, folder="images-scrapbook",
                               fmt=SAVE_FORMAT, quality=JPEG_QUALITY)

        final_pil.close()
        for im in cropped_for_page:
            if im: im.close()

        return (int(index_str), url)

    # ---------- streaming pipeline ----------
    def process_template(self, template_data: TemplateData) -> Dict:
        results = {"run_id": template_data.id, "pages": [], "errors": []}
        page_keys_sorted = sorted(template_data.frame_images.keys(), key=int)
        
        all_slots = [slot for k in page_keys_sorted for slot in template_data.frame_images[k].photo_slots]
        
        all_uploaded_bytes = self._load_many_bytes(template_data.uploaded_images)
        
        print(f"Melakukan crop untuk {len(all_slots)} slot...")
        all_cropped_photos = self._crop_photos_for_slots_parallel(all_uploaded_bytes, all_slots)
        print("✅ Semua foto berhasil di-crop.")
        
        photo_cursor = 0
        with ThreadPoolExecutor(max_workers=WORKERS_COMPOSITE) as ex:
            futs = []
            for k in page_keys_sorted:
                page_config = template_data.frame_images[k]
                slots_on_page = page_config.photo_slots
                num_slots = len(slots_on_page)
                
                photos_for_this_page = all_cropped_photos[photo_cursor : photo_cursor + num_slots]
                photo_cursor += num_slots
                
                frame_bytes = self._load_image_bytes(page_config.frame_image_path)
                
                fut = ex.submit(
                    self._composite_one_page,
                    k, frame_bytes, photos_for_this_page, slots_on_page,
                    page_config.svg_paths, template_data.id
                )
                futs.append(fut)

            for fut in as_completed(futs):
                try:
                    page_idx, url = fut.result()
                    results["pages"].append({"index": page_idx, "url": url})
                except Exception as e:
                    print(f"Error: {traceback.format_exc()}")
                    results["errors"].append({"error": f"{type(e).__name__}: {e}"})

        results["pages"].sort(key=lambda x: x["index"])
        return results
