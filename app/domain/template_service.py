import os
import base64
import requests
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.delivery.schemas.body import TemplateData, PageData, Slot
from app.domain.yolo_processor import YOLOProcessor
from app.infrastructure.cv.image_process import insert_photos_with_svg_mask_new

REQUEST_TIMEOUT = 15

WORKERS_IO = 8          # parallel downloads / file reads
WORKERS_CROP = 6        # parallel YOLO crops (CPU-bound but Torch/CV2 release GIL a lot)
WORKERS_COMPOSITE = 4   # parallel page compositing

SAVE_FORMAT: str = "jpg"
JPEG_QUALITY: int = 88

class TemplateService:
    def __init__(self, yolo: YOLOProcessor):
        self.yolo = yolo
        os.makedirs("outputs", exist_ok=True)

    def _load_image_bytes(self, src: str) -> bytes:
        """URL, local path, data URL, or raw base64 → bytes (single)."""
        if src.startswith("http://") or src.startswith("https://"):
            r = requests.get(src, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.content
        if os.path.isfile(src):
            with open(src, "rb") as f:
                return f.read()
        if src.startswith("data:image"):
            _, encoded = src.split(",", 1)
            return base64.b64decode(encoded + "===")  # tolerate missing padding
        return base64.b64decode(src + "===")          # tolerate missing padding

    def _decode_cv(self, b: bytes, imread_flag=cv2.IMREAD_COLOR):
        arr = np.frombuffer(b, np.uint8)
        img = cv2.imdecode(arr, imread_flag)
        if img is None:
            raise ValueError("Failed to decode image")
        return img

    def _load_many_bytes(self, sources: List[str]) -> List[bytes]:
        """Parallel load many images → bytes (keeps order)."""
        out = [None] * len(sources)
        with ThreadPoolExecutor(max_workers=WORKERS_IO) as ex:
            futs = {ex.submit(self._load_image_bytes, src): i for i, src in enumerate(sources)}
            for fut in as_completed(futs):
                i = futs[fut]
                out[i] = fut.result()
        return out


    def _crop_one(self, img_bytes: bytes, slot: Slot) -> Image.Image:
        img_cv = self._decode_cv(img_bytes, cv2.IMREAD_COLOR)
        cropped_cv = self.yolo.crop_person(img_cv, slot.w, slot.h)
        return Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))

    def _crop_photos_for_slots_parallel(self, images_bytes: List[bytes], slots: List[Slot]) -> List[Image.Image]:
        """Parallel YOLO crop for each (image, slot) pair (keeps order)."""
        if len(images_bytes) < len(slots):
            raise ValueError(f"Not enough images: need {len(slots)}, got {len(images_bytes)}")

        tasks = list(zip(images_bytes[:len(slots)], slots))
        out: List[Image.Image] = [None] * len(tasks)

        with ThreadPoolExecutor(max_workers=WORKERS_CROP) as ex:
            futs = {ex.submit(self._crop_one, img_b, slot): idx for idx, (img_b, slot) in enumerate(tasks)}
            for fut in as_completed(futs):
                idx = futs[fut]
                out[idx] = fut.result()
        return out

    def _composite_one_page(
        self,
        index_str: str,
        frame_bytes: bytes,
        cropped_for_page: List[Image.Image],
        slots: List[Slot],
        svg_groups: List[List[str]],
        out_name: str,
    ) -> Tuple[int, str]:
        frame_cv = self._decode_cv(frame_bytes, cv2.IMREAD_COLOR)
        final_pil = insert_photos_with_svg_mask_new(
            frame_cv,
            cropped_for_page,
            [s.model_dump() for s in slots],
            svg_groups
        )
        if final_pil is None:
            raise ValueError("Compositing failed")
        ext = SAVE_FORMAT.lower()
        out_path = os.path.join("image-scrapbook", f"{out_name}_page_{index_str}.{ext}")

        if ext == "jpg" or ext == "jpeg":
            # Convert to RGB, ensure no alpha
            rgb = final_pil.convert("RGB")
            rgb.save(out_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
            return (int(index_str), out_path)
        else:
            final_pil.save(out_path)  # PNG path if you really need alpha
            return (int(index_str), out_path)

    def process_template(self, template_data: TemplateData):
        out_name = template_data.name or template_data.id

        # Build global slot order and per-page mapping
        page_keys_sorted = sorted(template_data.frame_images.keys(), key=lambda k: int(k))
        all_slots: List[Slot] = []
        page_slots_map: Dict[str, List[Slot]] = {}
        frame_sources: List[str] = []
        for k in page_keys_sorted:
            page: PageData = template_data.frame_images[k]
            page_slots_map[k] = page.photo_slots
            all_slots.extend(page.photo_slots)
            frame_sources.append(page.frame_image_path)

        # Load uploads in parallel (bytes) + crop in parallel
        upload_bytes = self._load_many_bytes(template_data.uploaded_images)
        cropped_all = self._crop_photos_for_slots_parallel(upload_bytes, all_slots)

        # Load all frames in parallel (bytes)
        frame_bytes_list = self._load_many_bytes(frame_sources) 

        # Composite pages in parallel
        results = {"run_id": template_data.id, "pages": [], "errors": []}
        # slice cropped images per page
        cursor = 0
        jobs = []

        with ThreadPoolExecutor(max_workers=WORKERS_COMPOSITE) as ex:
            for idx, k in enumerate(page_keys_sorted):
                slots = page_slots_map[k]
                n = len(slots)
                cropped_for_page = cropped_all[cursor:cursor+n]
                cursor += n
                frame_b = frame_bytes_list[idx]
                svg_groups = template_data.frame_images[k].svg_paths
                # schedule
                jobs.append((k, ex.submit(
                    self._composite_one_page, k, frame_b, cropped_for_page, slots, svg_groups, out_name
                )))

            for k, fut in jobs:
                try:
                    page_idx, out_path = fut.result()
                    results["pages"].append({"index": page_idx, "output_path": out_path})
                except Exception as e:
                    results["errors"].append({"index": int(k), "error": str(e)})

        # sort pages by index for tidy output
        results["pages"].sort(key=lambda x: x["index"])
        return results
