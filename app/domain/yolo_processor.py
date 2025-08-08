from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import cv2
import numpy as np
import base64
import torch
from app.infrastructure.cv.image_process import smart_crop_with_yolo
import os
from app.config.settings import settings
import torch.nn.modules.container as container

# Configure safe globals for YOLO model loading
torch.serialization.add_safe_globals([
    tasks.DetectionModel,
    container.Sequential
])

class YOLOProcessor:
    """Handles YOLO model operations with singleton pattern for model instance"""
    _model_instance = None

    def __init__(self):
        if YOLOProcessor._model_instance is None:
            YOLOProcessor._model_instance = self._load_model()
        self.model = YOLOProcessor._model_instance

    def _load_model(self):
        """Loads or downloads YOLO model if not present"""
        os.makedirs("models", exist_ok=True)
        model_path = settings.YOLO_MODEL_PATH

        if not os.path.exists(model_path):
            print(f"[YOLOProcessor] Model not found, downloading to {model_path}...")
            self._download_model(model_path)

        print(f"[YOLOProcessor] Loading YOLO model from {model_path}...")
        return YOLO(model_path)

    def _download_model(self, dest_path: str):
        """Downloads YOLO model from official repository"""
        import urllib.request
        url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt"
        urllib.request.urlretrieve(url, dest_path)
        print(f"[YOLOProcessor] Download complete: {dest_path}")

    def _decode_image(self, img_data: str):
        """Converts base64 or file path to OpenCV image"""
        if img_data.startswith("data:image"):
            _, encoded = img_data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
        else:
            with open(img_data, "rb") as f:
                img_bytes = f.read()

        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def crop_person(self, image_cv, target_w, target_h):
        """Crops person from image using YOLO detection"""
        return smart_crop_with_yolo(self.model, image_cv, target_w, target_h)

    def process_images(self, images_base64, target_w, target_h):
        """Batch processes multiple images with YOLO detection"""
        processed_images = []
        for img_str in images_base64:
            image_cv = self._decode_image(img_str)
            cropped = self.crop_person(image_cv, target_w, target_h)
            processed_images.append(cropped)
        return processed_images
