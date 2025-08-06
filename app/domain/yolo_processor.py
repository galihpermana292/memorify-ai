from ultralytics import YOLO
from app.config import settings

class YOLOProcessor:
    def __init__(self):
        self.model = YOLO(settings.YOLO_MODEL_PATH)

    def crop_person(self, image_cv, target_w, target_h):
        from app.infrastructure.cv.image_utils import smart_crop_with_yolo
        return smart_crop_with_yolo(self.model, image_cv, target_w, target_h)
