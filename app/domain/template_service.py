import os
import random
import base64
import requests
import numpy as np
import cv2
from PIL import Image
from app.delivery.schemas.body import TemplateData
from app.domain.yolo_processor import YOLOProcessor
from app.infrastructure.cv.image_process import insert_photos_with_svg_mask
from app.infrastructure.cv.image_process import insert_photos_with_svg_mask_new

class TemplateService:
    def __init__(self, yolo: YOLOProcessor):
        self.yolo = yolo

    def _load_image_bytes(self, img_source):
        """
        Converts a URL, base64 string, or bytes into raw image bytes.
        """
        if isinstance(img_source, bytes):
            return img_source
        elif isinstance(img_source, str):
            if img_source.startswith("http://") or img_source.startswith("https://"):
                resp = requests.get(img_source)
                resp.raise_for_status()
                return resp.content
            else:
                # Assume it's base64
                return base64.b64decode(img_source)
        else:
            raise ValueError("Unsupported image format")

    async def process_template(self, template_data: TemplateData):
        """
        Processes template data provided by the API.
        Args:
            template_data: TemplateData object containing:
                - frame_images: dict of page numbers to frame image data
                - svg_files: list of SVG mask files
                - photo_slots: list of dicts with photo slot dimensions
                - uploaded_images: list of image sources (URL/base64/bytes)
        Returns:
            list: Paths to processed output images
        """
        # Validate input data
        if not template_data.frame_images or not template_data.svg_files or not template_data.photo_slots:
            raise ValueError("Missing required template data")

        # Ensure we have enough images
        if len(template_data.uploaded_images) < len(template_data.photo_slots):
            raise ValueError(f"Not enough images provided. Need {len(template_data.photo_slots)}, got {len(template_data.uploaded_images)}")

        # Process images
        cropped_photos = []
        for img_source, slot in zip(template_data.uploaded_images[:len(template_data.photo_slots)], 
                                template_data.photo_slots):
            try:
                img_bytes = self._load_image_bytes(img_source)
                image_array = np.frombuffer(img_bytes, np.uint8)
                raw_image_cv = cv2.imdecode(image_array, cv2.IMREAD_COLOR)  # Changed to COLOR
                if raw_image_cv is None:
                    raise ValueError(f"Failed to decode image")
                    
                cropped = self.yolo.crop_person(raw_image_cv, slot["w"], slot["h"])
                cropped_photos.append(Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)))
            except Exception as e:
                raise ValueError(f"Error processing image: {str(e)}")

        # Process frames
        results = []
        os.makedirs("outputs", exist_ok=True)
        
        for page_num, frame_data in template_data.frame_images.items():
            try:
                # Get the URL from the input data
                frame_url = frame_data["frame_image_path"]

                # --- FIX STARTS HERE ---

                # 1. Download the frame image from the URL
                frame_bytes = self._load_image_bytes(frame_url)
                
                # 2. Decode the downloaded bytes into an image object
                frame_array = np.frombuffer(frame_bytes, np.uint8)
                frame_image_cv = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                if frame_image_cv is None:
                    raise ValueError(f"Failed to decode frame image from URL: {frame_url}")

                # --- FIX ENDS HERE ---
                
                svg_path_groups = [[svg] for svg in template_data.svg_files]

                # Now, pass the loaded image object (not the path) to your function
                # NOTE: You may need to adjust insert_photos_with_svg_mask to accept an image object
                # instead of a file path for the first argument.
                final_image = insert_photos_with_svg_mask_new(
                    frame_image_cv,  # <-- Pass the loaded image object here
                    cropped_photos,
                    template_data.photo_slots,
                    svg_path_groups
                )
                
                output_path = os.path.join("outputs", f"output_page_{page_num}.png")
                
                # Convert back to PIL Image to save, if final_image is a NumPy array
                final_image.save(output_path)
                
                results.append(output_path)
            except Exception as e:
                # The error message was already good, so we keep it
                raise ValueError(f"Error processing frame {page_num}: {str(e)}")

            return results
        