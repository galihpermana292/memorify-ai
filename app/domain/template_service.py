# app/domain/template_service.py
import threading
import logging
from typing import List, Dict, Tuple
import traceback, base64, os, io, time
import numpy as np, cv2
from PIL import Image
import asyncio
import aiohttp
import psutil
import aiofiles
import gc
from concurrent.futures import ThreadPoolExecutor

from app.delivery.schemas.body import TemplateData
from app.infrastructure.cv import image_process
from app.infrastructure.cloudinary.upload_file import upload_pil_image
from app.domain.yolo_processor import YOLOProcessor

# --- KONFIGURASI ---
REQUEST_TIMEOUT = 30
SAVE_FORMAT = "jpeg"
JPEG_QUALITY = 75

# --- PENGATURAN LOGGER ---
# Menginisialisasi logger khusus untuk modul ini untuk menghindari konflik
# dan memungkinkan konfigurasi yang lebih terperinci.
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False # Mencegah log ganda ke root logger

class TemplateService:
    def __init__(self, yolo: YOLOProcessor, cpu_executor: ThreadPoolExecutor, io_executor: ThreadPoolExecutor):
        self.yolo_processor = yolo
        self.yolo_lock = threading.Lock()
        self.cpu_executor = cpu_executor
        self.io_executor = io_executor

    async def _load_image_bytes_async(self, src: str, session: aiohttp.ClientSession) -> bytes:
        try:
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
        except Exception as e:
            logger.warning(f"Gagal memuat gambar dari sumber '{src[:70]}...': {type(e).__name__}")
            return None

    async def _load_many_bytes_async(self, sources: List[str]) -> List[bytes]:
        async with aiohttp.ClientSession() as session:
            tasks = [self._load_image_bytes_async(src, session) for src in sources]
            return await asyncio.gather(*tasks)

    def _decode_cv(self, b: bytes, imread_flag=cv2.IMREAD_COLOR, max_side=800):
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

    def _crop_one(self, img_bytes: bytes, slot) -> Image.Image:
        if img_bytes is None: return None
        img_cv = self._decode_cv(img_bytes)
        if img_cv is None:
            logger.warning("Gagal decode gambar, proses crop dilewati.")
            return None
        
        slot_dict = slot.model_dump()
        cropped_pil = None
        try:
            with self.yolo_lock:
                cropped_pil = self.yolo_processor.crop_person(img_cv, int(slot_dict['w']), int(slot_dict['h']))
        finally:
            del img_cv
            
        if cropped_pil is None:
            logger.warning("Cropping gagal, mengembalikan None.")
            return None
            
        del img_cv
        return cropped_pil

    async def _crop_photos_for_slots_parallel_async(self, images_bytes: List[bytes], slots) -> List[Image.Image]:
        num_slots = len(slots)
        valid_images = [b for b in images_bytes if b is not None]
        num_valid_images = len(valid_images)

        if num_valid_images == 0:
            logger.warning("Tidak ada gambar pengguna yang valid untuk di-crop.")
            return [None] * num_slots

        images_to_process = []
        if num_valid_images < num_slots:
            logger.info(f"Menggunakan {num_valid_images} gambar untuk {num_slots} slot (duplikasi diterapkan).")
            for i in range(num_slots):
                images_to_process.append(valid_images[i % num_valid_images])
        else:
            images_to_process = valid_images[:num_slots]

        tasks = list(zip(images_to_process[:num_slots], slots))
        loop = asyncio.get_running_loop()
        futures = [loop.run_in_executor(self.cpu_executor, self._crop_one, img_b, slot) for img_b, slot in tasks]
        return await asyncio.gather(*futures)

    def _upload_static_page(self, index_str: str, frame_bytes: bytes, out_id: str, master_start_time: float) -> Tuple[int, str, float]:
        if frame_bytes is None: raise ValueError(f"Frame untuk halaman statis {index_str} tidak bisa dimuat.")
        pil_img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        public_id = f"{out_id}_page_{index_str}"
        
        start_time = time.perf_counter()
        url = upload_pil_image(pil_img, public_id=public_id, folder="images-scrapbook", fmt=SAVE_FORMAT, quality=JPEG_QUALITY)
        end_time = time.perf_counter()
        
        elapsed_since_start = end_time - master_start_time
        logger.info(f"Halaman Statis [{index_str}]: Upload selesai (T+{elapsed_since_start:.2f}s).")
        pil_img.close()
        return (int(index_str), url, end_time - start_time)

    def _composite_one_page(self, index_str: str, frame_bytes: bytes, cropped_for_page: List[Image.Image], slots, svg_groups, out_id: str, master_start_time: float) -> Tuple[int, str, float]:
        if frame_bytes is None: raise ValueError(f"Frame untuk halaman {index_str} tidak bisa dimuat.")
        frame_pil = Image.open(io.BytesIO(frame_bytes)).convert("RGBA")
        
        valid_cropped_photos = [photo for photo in cropped_for_page if photo]
        if not valid_cropped_photos:
            logger.warning(f"Halaman [{index_str}]: Tidak ada foto valid, pembuatan halaman dilewati.")
            return (int(index_str), "", 0.0)

        slots_as_dicts = [s.model_dump() for s in slots]
        final_pil = image_process.insert_photos_with_svg_mask(frame_pil, cropped_for_page, slots_as_dicts, svg_groups)
        del frame_pil

        if final_pil is None:
            for im in cropped_for_page:
                if im: im.close()
            raise ValueError(f"Proses compositing untuk halaman {index_str} gagal.")
        
        public_id = f"{out_id}_page_{index_str}"
        start_time = time.perf_counter()
        url = upload_pil_image(final_pil.convert('RGB'), public_id=public_id, folder="images-scrapbook", fmt=SAVE_FORMAT, quality=JPEG_QUALITY)
        end_time = time.perf_counter()
        
        elapsed_since_start = end_time - master_start_time
        logger.info(f"Halaman [{index_str}]: Composite & Upload selesai (T+{elapsed_since_start:.2f}s).")
        
        final_pil.close()
        for im in cropped_for_page:
            if im: im.close()
        return (int(index_str), url, end_time - start_time)

    async def process_template(self, template_data: TemplateData) -> Dict:
        run_id = template_data.id
        logger.info(f"=== START PROCESSING Run ID: {run_id} ===")
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory usage at start: {memory_mb:.1f}MB for Run ID: {run_id}")
        except Exception as mem_error:
            logger.warning(f"Could not get memory info: {mem_error}")
        
        overall_start_time = time.perf_counter()
        
        try:
            results = {"run_id": run_id, "pages": [], "errors": []}
            page_keys_sorted = sorted(template_data.frame_images.keys(), key=int)
            
            pages_with_slots = {k: v for k, v in template_data.frame_images.items() if v.photo_slots}
            all_slots = [slot for k in sorted(pages_with_slots.keys()) for slot in pages_with_slots[k].photo_slots]
            
            # TAHAP 1: Download Aset
            logger.info(f"Tahap 1/4: Mengunduh {len(template_data.uploaded_images)} gambar pengguna dan {len(page_keys_sorted)} frame.")
            user_img_sources = template_data.uploaded_images
            frame_img_sources = [template_data.frame_images[k].frame_image_path for k in page_keys_sorted]
            
            all_user_bytes, all_frame_bytes_list = await asyncio.gather(
                self._load_many_bytes_async(user_img_sources),
                self._load_many_bytes_async(frame_img_sources)
            )
            all_frame_bytes_dict = dict(zip(page_keys_sorted, all_frame_bytes_list))
            logger.info(f"Tahap 1/4: Download selesai untuk Run ID: {run_id}")
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory after download: {memory_mb:.1f}MB for Run ID: {run_id}")
            
            # TAHAP 2: Smart Cropping
            all_cropped_photos = []
            crop_duration = 0
            if pages_with_slots:
                logger.info(f"Tahap 2/4: Melakukan smart-cropping untuk {len(all_slots)} slot.")
                crop_start_time = time.perf_counter()
                all_cropped_photos = await self._crop_photos_for_slots_parallel_async(all_user_bytes, all_slots)
                crop_duration = time.perf_counter() - crop_start_time
                logger.info(f"Tahap 2/4: Cropping selesai dalam {crop_duration:.2f} detik untuk Run ID: {run_id}")
                memory_mb = process.memory_info().rss / 1024 / 1024
                logger.info(f"Memory after cropping: {memory_mb:.1f}MB for Run ID: {run_id}")
            else:
                logger.info(f"Tahap 2/4: Tidak ada slot foto, tahap cropping dilewati untuk Run ID: {run_id}")
                
            # TAHAP 3: Compositing & Upload Paralel
            logger.info(f"Tahap 3/4: Memulai compositing dan upload paralel untuk Run ID: {run_id}")
            loop = asyncio.get_running_loop()
            all_futures = []
            upload_master_start_time = time.perf_counter()
            
            # Create futures
            photo_cursor = 0
            for k in sorted(template_data.frame_images.keys(), key=int):
                page_config = template_data.frame_images[k]
                if page_config.photo_slots:
                    num_slots = len(page_config.photo_slots)
                    photos_for_this_page = all_cropped_photos[photo_cursor : photo_cursor + num_slots]
                    photo_cursor += num_slots
                    fut = loop.run_in_executor(
                        self.cpu_executor, 
                        self._composite_one_page, 
                        k, 
                        all_frame_bytes_dict.get(k), 
                        photos_for_this_page, 
                        page_config.photo_slots, 
                        page_config.svg_paths, 
                        run_id, 
                        upload_master_start_time
                    )
                else:
                    fut = loop.run_in_executor(
                        self.io_executor, 
                        self._upload_static_page, 
                        k, 
                        all_frame_bytes_dict.get(k), 
                        run_id, 
                        upload_master_start_time
                    )
                all_futures.append(fut)
            
            logger.info(f"Created {len(all_futures)} futures for Run ID: {run_id}")
            
            # Process futures with timeout and better error handling
            total_upload_work_time = 0.0
            completed_count = 0
            failed_count = 0
            
            try:
                # Add overall timeout for all futures
                async with asyncio.timeout(300):  # 5 minute timeout for all processing
                    for fut in asyncio.as_completed(all_futures):
                        try:
                            page_idx, url, duration = await fut
                            total_upload_work_time += duration
                            completed_count += 1
                            
                            if url:
                                results["pages"].append({"index": page_idx, "url": url})
                                logger.info(f"Page {page_idx} completed successfully for Run ID: {run_id} ({completed_count}/{len(all_futures)})")
                            else:
                                logger.warning(f"Page {page_idx} completed but no URL returned for Run ID: {run_id}")
                                
                        except Exception as e:
                            failed_count += 1
                            logger.error(f"Future failed for Run ID {run_id}: {e}\n{traceback.format_exc()}")
                            results["errors"].append({"error": f"{type(e).__name__}: {e}"})
            
            except asyncio.TimeoutError:
                logger.error(f"TIMEOUT: Processing exceeded 5 minutes for Run ID: {run_id}")
                # Cancel remaining futures
                for fut in all_futures:
                    if not fut.done():
                        fut.cancel()
                results["errors"].append({"error": "Processing timeout after 5 minutes"})
                
            logger.info(f"Futures completed: {completed_count}, failed: {failed_count}, total: {len(all_futures)} for Run ID: {run_id}")
            
            upload_wall_time = time.perf_counter() - upload_master_start_time
            logger.info(f"Tahap 3/4: Upload paralel selesai. Waktu proses: {upload_wall_time:.2f} detik untuk Run ID: {run_id}")
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory after upload: {memory_mb:.1f}MB for Run ID: {run_id}")
            
            # TAHAP 4: Finalisasi
            logger.info(f"Tahap 4/4: Finalisasi hasil untuk Run ID: {run_id}")
            results["pages"].sort(key=lambda x: x["index"])
            
            overall_duration = time.perf_counter() - overall_start_time
            
            # Cleanup
            try:
                del all_cropped_photos, all_frame_bytes_dict, all_user_bytes, all_futures, all_slots
                gc.collect()
            except Exception as cleanup_error:
                logger.warning(f"Cleanup error for Run ID {run_id}: {cleanup_error}")
            
            logger.info(f"=== COMPLETED PROCESSING Run ID: {run_id} dalam {overall_duration:.2f} detik ===")
            return results
            
        except Exception as e:
            logger.error(f"=== CRITICAL ERROR in process_template for Run ID {run_id}: {e}\n{traceback.format_exc()} ===")
            raise