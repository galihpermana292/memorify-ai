# app/domain/template_service.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import traceback, base64, os, io
import numpy as np, cv2
from PIL import Image
import asyncio
import aiohttp
import aiofiles
import gc

from app.delivery.schemas.body import TemplateData
from app.infrastructure.cv import image_process
from app.infrastructure.cloudinary.upload_file import upload_pil_image
from app.domain.yolo_processor import YOLOProcessor

# --- KONFIGURASI OPTIMASI ---
REQUEST_TIMEOUT = 30
SAVE_FORMAT = "jpeg"
JPEG_QUALITY = 88

class TemplateService:
    def __init__(self, yolo: YOLOProcessor):
        self.yolo_processor = yolo
        os.makedirs("outputs", exist_ok=True)

    # ---------- IO helpers (Async) ----------
    async def _load_image_bytes_async(self, src: str, session: aiohttp.ClientSession) -> bytes:
        if src.startswith(("http://", "https://")):
            async with session.get(src, timeout=REQUEST_TIMEOUT) as response:
                response.raise_for_status()
                return await response.read()
        if os.path.isfile(src):
            async with aiofiles.open(src, "rb") as f:
                return await f.read()
        if src.startswith("data:image"):
            _, encoded = src.split(",", 1)
            return base64.b64decode(encoded + "===")
        return base64.b64decode(src + "===")

    async def _load_many_bytes_async(self, sources: List[str]) -> List[bytes]:
        async with aiohttp.ClientSession() as session:
            tasks = [self._load_image_bytes_async(src, session) for src in sources]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            processed_results = []
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    error_type = type(res).__name__
                    print(f"⚠️ Gagal mengunduh gambar dari sumber '{sources[i]}': [{error_type}] - {res}")
                    processed_results.append(None)
                else:
                    processed_results.append(res)
            return processed_results

    def _decode_cv(self, b: bytes, imread_flag=cv2.IMREAD_COLOR, max_side=800):
        # ... (fungsi ini tidak berubah)
        if b is None: return None
        arr = np.frombuffer(b, np.uint8)
        img = cv2.imdecode(arr, imread_flag)
        if img is None: return None
        h, w = img.shape[:2]
        m = max(h, w)
        if m > max_side:
            scale = max_side / m
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        return img

    # ---------- crop/composite ----------
    def _crop_one(self, img_bytes: bytes, slot) -> Image.Image:
        # ... (fungsi ini tidak berubah)
        if img_bytes is None: return None
        img_cv = self._decode_cv(img_bytes) 
        if img_cv is None: 
            print("⚠️ Warning: Gagal men-decode gambar, proses crop dilewati.")
            return None
        slot_dict = slot.model_dump()
        cropped_cv = self.yolo_processor.crop_person(img_cv, int(slot_dict['w']), int(slot_dict['h']))
        if cropped_cv is None:
            print("⚠️ Warning: Cropping gagal, mengembalikan None.")
            return None
        pil_img = Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))
        del img_cv, cropped_cv
        return pil_img

    async def _crop_photos_for_slots_parallel_async(self, images_bytes: List[bytes], slots) -> List[Image.Image]:
        # ... (fungsi ini tidak berubah)
        num_slots = len(slots)
        num_images = len(images_bytes)
        images_to_process = list(images_bytes) 
        if num_images < num_slots:
            print(f"⚠️ Jumlah gambar ({num_images}) lebih sedikit dari slot ({num_slots}). Gambar akan diduplikasi.")
            for i in range(num_slots - num_images):
                images_to_process.append(images_bytes[i % num_images])
        tasks = list(zip(images_to_process[:num_slots], slots))
        loop = asyncio.get_running_loop()
        futures = [loop.run_in_executor(None, self._crop_one, img_b, slot) for img_b, slot in tasks]
        out = await asyncio.gather(*futures)
        return out

    # --- FUNGSI BARU UNTUK HALAMAN STATIS ---
    def _upload_static_page(self, index_str: str, frame_bytes: bytes, out_id: str) -> Tuple[int, str]:
        """Fungsi ini hanya meng-upload gambar frame apa adanya."""
        if frame_bytes is None:
            raise ValueError(f"Frame untuk halaman statis {index_str} tidak bisa dimuat.")

        # Buka sebagai PIL untuk memastikan formatnya benar dan mengonversi ke RGB
        pil_img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        
        public_id = f"{out_id}_page_{index_str}"
        url = upload_pil_image(pil_img, public_id=public_id, folder="images-scrapbook",
                               fmt=SAVE_FORMAT, quality=JPEG_QUALITY)
        pil_img.close()
        return (int(index_str), url)

    def _composite_one_page(
        self, index_str: str, frame_bytes: bytes,
        cropped_for_page: List[Image.Image], slots, svg_groups, out_id: str
    ) -> Tuple[int, str]:
        # ... (fungsi ini tidak berubah, tetapi sekarang hanya akan dipanggil untuk halaman dengan slot)
        if frame_bytes is None:
            raise ValueError(f"Frame untuk halaman {index_str} tidak bisa dimuat.")
        frame_pil = Image.open(io.BytesIO(frame_bytes)).convert("RGBA")
        valid_cropped_photos = []
        valid_slots = []
        for i, photo in enumerate(cropped_for_page):
            if photo:
                valid_cropped_photos.append(photo)
                valid_slots.append(slots[i])

        if not valid_cropped_photos:
              print(f"⚠️ Tidak ada foto valid untuk halaman {index_str}, halaman tidak dibuat.")
              return (int(index_str), "")

        slots_as_dicts = [s.model_dump() for s in valid_slots]
        final_pil = image_process.insert_photos_with_svg_mask(frame_pil, valid_cropped_photos, slots_as_dicts, svg_groups)
        del frame_pil

        if final_pil is None:
            for im in cropped_for_page:
                if im: im.close()
            raise ValueError("Compositing failed")
        
        public_id = f"{out_id}_page_{index_str}"
        url = upload_pil_image(final_pil.convert('RGB'), public_id=public_id, folder="images-scrapbook",
                               fmt=SAVE_FORMAT, quality=JPEG_QUALITY)

        final_pil.close()
        for im in cropped_for_page:
            if im: im.close()

        return (int(index_str), url)

    # ---------- pipeline (Async) - LOGIKA DIPERBARUI ----------
    async def process_template(self, template_data: TemplateData) -> Dict:
        results = {"run_id": template_data.id, "pages": [], "errors": []}
        page_keys_sorted = sorted(template_data.frame_images.keys(), key=int)
        
        # Pisahkan halaman yang punya slot dan yang tidak
        pages_with_slots = {k: v for k, v in template_data.frame_images.items() if v.photo_slots}
        pages_without_slots = {k: v for k, v in template_data.frame_images.items() if not v.photo_slots}

        all_slots = [slot for k in sorted(pages_with_slots.keys()) for slot in pages_with_slots[k].photo_slots]
        
        # Lakukan proses berat (download & crop foto) hanya jika ada halaman dengan slot
        all_cropped_photos = []
        if pages_with_slots:
            print("Mengunduh semua gambar pengguna secara asinkron...")
            all_uploaded_bytes = await self._load_many_bytes_async(template_data.uploaded_images)
            print("✅ Semua gambar pengguna berhasil diunduh.")

            print(f"Melakukan crop untuk {len(all_slots)} slot...")
            all_cropped_photos = await self._crop_photos_for_slots_parallel_async(all_uploaded_bytes, all_slots)
            print("✅ Semua foto berhasil di-crop.")
        
        photo_cursor = 0
        loop = asyncio.get_running_loop()
        all_futures = []
        
        # Download semua frame sekaligus
        all_frame_sources = [template_data.frame_images[k].frame_image_path for k in page_keys_sorted]
        all_frame_bytes_dict = dict(zip(page_keys_sorted, await self._load_many_bytes_async(all_frame_sources)))
        
        # Proses halaman yang DENGAN slot
        for k in sorted(pages_with_slots.keys()):
            page_config = pages_with_slots[k]
            num_slots = len(page_config.photo_slots)
            
            photos_for_this_page = all_cropped_photos[photo_cursor : photo_cursor + num_slots]
            photo_cursor += num_slots
            
            frame_bytes = all_frame_bytes_dict.get(k)
            
            fut = loop.run_in_executor(
                None, self._composite_one_page,
                k, frame_bytes, photos_for_this_page, page_config.photo_slots,
                page_config.svg_paths, template_data.id
            )
            all_futures.append(fut)

        # Proses halaman yang TANPA slot
        for k in sorted(pages_without_slots.keys()):
            frame_bytes = all_frame_bytes_dict.get(k)
            fut = loop.run_in_executor(
                None, self._upload_static_page, k, frame_bytes, template_data.id
            )
            all_futures.append(fut)
        
        # Kumpulkan semua hasil
        for fut in asyncio.as_completed(all_futures):
            try:
                page_idx, url = await fut
                if url:
                    results["pages"].append({"index": page_idx, "url": url})
            except Exception as e:
                print(f"Error: {traceback.format_exc()}")
                results["errors"].append({"error": f"{type(e).__name__}: {e}"})

        results["pages"].sort(key=lambda x: x["index"])
        
        # --- BLOK PEMBERSIHAN MEMORI ---
        del all_cropped_photos
        del all_frame_bytes_dict
        del all_slots
        del all_futures
        if 'all_uploaded_bytes' in locals():
            del all_uploaded_bytes
        
        gc.collect()
        print("🗑️ Pembersihan memori eksplisit (GC) selesai.")
        
        return results