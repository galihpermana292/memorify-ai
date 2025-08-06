from sqlalchemy.ext.asyncio import AsyncSession
from app.infrastructure.database.models import Template
from app.domain.yolo_processor import YOLOProcessor
from app.infrastructure.cv.image_process import insert_photos_with_svg_mask
from PIL import Image
import numpy as np
import cv2
import random

class TemplateService:
    def __init__(self, db: AsyncSession, yolo: YOLOProcessor):
        self.db = db
        self.yolo = yolo

    async def process_template(self, template_id: str, uploaded_images: list):
        template = await self.db.get(Template, template_id)
        if not template:
            raise ValueError("Template not found")

        data = template.frame_data
        photo_slots = [slot for page in data.values() for slot in page.get("photo_slots", [])]
        total_slots_needed = len(photo_slots)

        if len(uploaded_images) > total_slots_needed:
            uploaded_images = random.sample(uploaded_images, total_slots_needed)

        cropped_photos = []
        for img_bytes, slot in zip(uploaded_images, photo_slots):
            image_array = np.frombuffer(img_bytes, np.uint8)
            raw_image_cv = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
            cropped = self.yolo.crop_person(raw_image_cv, slot["w"], slot["h"])
            cropped_photos.append(Image.fromarray(cropped))

        results = []
        for page_num, page_data in data.items():
            frame_path = page_data["frame_image_path"]
            svg_paths = page_data.get("svg_paths", [])
            slots = page_data["photo_slots"]

            final_image = insert_photos_with_svg_mask(frame_path, cropped_photos, slots, svg_paths)
            output_path = f"output_page_{page_num}.png"
            final_image.save(output_path)
            results.append(output_path)

        return results
