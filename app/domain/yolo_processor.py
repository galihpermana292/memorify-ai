# app/domain/yolo_processor.py
import torch
import os
import shutil
from ultralytics import YOLO

class YOLOProcessor:
    _model_instance = None
    
    def warmup(self, imgsz: int = 320):
        try:
            from PIL import Image
            _ = self.model.predict(Image.new("RGB", (imgsz, imgsz)), imgsz=imgsz, verbose=False, device=getattr(self, "_device", "cpu"))
        except Exception:
            pass

    def __init__(self):
        if YOLOProcessor._model_instance is None:
            YOLOProcessor._model_instance = self._load_optimized_model()
        self.model = YOLOProcessor._model_instance

    def _load_optimized_model(self):
        from app.config.settings import settings
        os.makedirs("models", exist_ok=True)
        
        base_model_path = settings.YOLO_MODEL_PATH
        model_name = os.path.splitext(os.path.basename(base_model_path))[0]
        
        openvino_model_dir = base_model_path.replace('.pt', '_int8_openvino_model')
        # Cek untuk file .xml DAN .bin. Keduanya wajib ada.
        expected_xml_file = os.path.join(openvino_model_dir, f"{model_name}.xml")
        expected_bin_file = os.path.join(openvino_model_dir, f"{model_name}.bin")
        
        # JIKA MODEL TIDAK LENGKAP (Logika Regenerasi)
        if not (os.path.exists(expected_xml_file) and os.path.exists(expected_bin_file)):
            print("Model OpenVINO tidak lengkap atau tidak ditemukan. Meregenerasi model...")
            
            if os.path.exists(openvino_model_dir):
                print(f"Menghapus direktori model yang tidak lengkap: {openvino_model_dir}")
                shutil.rmtree(openvino_model_dir)

            if not os.path.exists(base_model_path):
                print(f"Model dasar {base_model_path} tidak ditemukan. Mengunduh...")
                YOLO(base_model_path)

            print("Memuat model PyTorch untuk konversi...")
            model_untuk_ekspor = YOLO(base_model_path)
            
            print("Mengekspor ke format OpenVINO dengan kuantisasi INT8...")
            
            try:
                model_untuk_ekspor.export(format='openvino', int8=True, data='coco128.yaml', half=False)
                print("✅ Model berhasil diekspor.")
            except Exception as e:
                print(f"❌ GAGAL mengekspor model ke OpenVINO: {e}")
                raise e

            print("Langsung menggunakan instance model yang baru saja diekspor untuk menghindari bug.")
            final_model = model_untuk_ekspor

        # JIKA MODEL SUDAH ADA (Logika Pemuatan Normal)
        else:
            print(f"Memuat model OpenVINO yang sudah ada dari {openvino_model_dir}...")
            try:
                final_model = YOLO(os.path.normpath(openvino_model_dir), task='detect')
            except Exception as e:
                print(f"Gagal memuat model yang sudah ada: {e}. Menghapus model korup.")
                shutil.rmtree(openvino_model_dir)
                raise RuntimeError("Gagal memuat model OpenVINO. Harap restart aplikasi untuk meregenerasi.") from e
        
        # Konfigurasi properti OpenVINO pada model final
        if hasattr(final_model.model, 'core'):
            cpu_count = os.cpu_count() or 1
            final_model.model.core.set_property({'INFERENCE_NUM_THREADS': cpu_count})
            print(f"Jumlah thread untuk inferensi OpenVINO diatur ke {cpu_count}")

        return final_model

    def crop_person(self, image_cv, target_w, target_h):
        from app.infrastructure.cv.image_process import smart_crop_for_template
        return smart_crop_for_template(self.model, image_cv, target_w, target_h)

    def _download_base_model(self, dest_path: str):
        # Gunakan YOLO untuk mengunduh model standar, ini lebih robust
        print(f"Mengunduh dan menyimpan model ke {dest_path}")
        YOLO(dest_path) # Cukup inisialisasi untuk men-trigger download