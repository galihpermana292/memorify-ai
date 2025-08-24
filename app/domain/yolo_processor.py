# app/domain/yolo_processor.py
import os
import logging
from ultralytics import YOLO
from app.config.settings import settings

# --- Pengaturan Logger ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [YOLO] %(message)s', '%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

class YOLOProcessor:
    _model_instance = None

    def __init__(self):
        if YOLOProcessor._model_instance is None:
            YOLOProcessor._model_instance = self._load_model()
        self.model = YOLOProcessor._model_instance

    def _load_model(self):
        base_model_path = settings.YOLO_MODEL_PATH
        openvino_model_dir = base_model_path.replace('.pt', '_int8_openvino_model')

        if not os.path.exists(openvino_model_dir):
            logger.error(f"Direktori model OpenVINO tidak ditemukan di: {openvino_model_dir}")
            raise FileNotFoundError(f"Model OpenVINO yang diharapkan tidak ada. Harap ekspor model terlebih dahulu.")

        try:
            logger.info(f"Memuat model OpenVINO dari: {openvino_model_dir}")
            # Memuat model langsung dari folder OpenVINO
            final_model = YOLO(os.path.normpath(openvino_model_dir), task='detect')
            
            # Mengatur properti OpenVINO untuk optimasi CPU
            if hasattr(final_model.model, 'core'):
                cpu_count = os.cpu_count() or 1
                final_model.model.core.set_property({'INFERENCE_NUM_THREADS': cpu_count})
                logger.info(f"Inferensi OpenVINO diatur untuk menggunakan {cpu_count} thread CPU.")
            
            logger.info("Model OpenVINO berhasil dimuat dan dikonfigurasi.")
            return final_model

        except Exception as e:
            logger.error(f"Gagal memuat model OpenVINO dari {openvino_model_dir}: {e}", exc_info=True)
            raise RuntimeError("Terjadi kesalahan kritis saat memuat model OpenVINO.") from e

    def crop_person(self, image_cv, target_w, target_h):
        from app.infrastructure.cv.image_process import smart_crop_for_template
        return smart_crop_for_template(self.model, image_cv, target_w, target_h)

    def warmup(self, imgsz: int = 416):
        try:
            from PIL import Image
            logger.info(f"Menjalankan pemanasan model dengan ukuran input {imgsz}x{imgsz}...")
            # Menjalankan prediksi pada gambar kosong
            _ = self.model.predict(Image.new("RGB", (imgsz, imgsz)), imgsz=imgsz, verbose=False)
            logger.info("Pemanasan model berhasil.")
        except Exception as e:
            logger.error(f"Pemanasan model gagal: {e}", exc_info=True)