# app/domain/yolo_processor.py
from ultralytics import YOLO
import torch, os
import ultralytics.nn.tasks as tasks
import torch.nn.modules.container as container
torch.serialization.add_safe_globals([tasks.DetectionModel, container.Sequential])

class YOLOProcessor:
    _model_instance = None

    def __init__(self):
        if YOLOProcessor._model_instance is None:
            YOLOProcessor._model_instance = self._load_model()
        self.model = YOLOProcessor._model_instance

    def _load_model(self):
        from app.config.settings import settings
        os.makedirs("models", exist_ok=True)
        model_path = settings.YOLO_MODEL_PATH

        if not os.path.exists(model_path):
            self._download_model(model_path)

        model = YOLO(model_path)
        # Optional: move to GPU & half precision if available
        if torch.cuda.is_available():
            model.to('cuda')
            try:
                model.model.half()
            except Exception:
                pass
        try:
            model.fuse()
        except Exception:
            pass
        return model

    def _download_model(self, dest_path: str):
        import urllib.request
        url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
        urllib.request.urlretrieve(url, dest_path)

    def crop_person(self, image_cv, target_w, target_h):
        # Expect BGR coming in
        from app.infrastructure.cv.image_process import smart_crop_for_template
        return smart_crop_for_template(self.model, image_cv, target_w, target_h)
