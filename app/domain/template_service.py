# app/domain/template_service.py
import logging
from typing import List, Dict, Tuple, Optional
import traceback, base64, os, io, time, gc
import numpy as np, cv2
from PIL import Image
import asyncio
import aiohttp
import aiofiles
import psutil
from concurrent.futures import ThreadPoolExecutor
from fastapi import Request  # <-- add

from app.delivery.schemas.body import TemplateData
from app.infrastructure.cv import image_process
from app.infrastructure.cloudinary.upload_file import upload_pil_image
from app.domain.yolo_processor import YOLOProcessor

REQUEST_TIMEOUT = 30
SAVE_FORMAT = "jpeg"
JPEG_QUALITY = 75

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

class TemplateService:
    def __init__(self, yolo: YOLOProcessor, cpu_executor: ThreadPoolExecutor, io_executor: ThreadPoolExecutor):
        # Reuse the singleton YOLO model instance supplied from main.py
        self.yolo = yolo
        self.cpu_executor = cpu_executor
        self.io_executor = io_executor

    async def warmup(self, imgsz: int = 416) -> None:
        # warm model on a thread so it doesn't block the loop
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.cpu_executor, self.yolo.warmup, imgsz)

    async def _load_image_bytes_async(self, src: str, session: aiohttp.ClientSession) -> Optional[bytes]:
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

    async def _load_many_bytes_async(self, sources: List[str]) -> List[Optional[bytes]]:
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

    def _crop_one(self, img_bytes: Optional[bytes], slot) -> Optional[Image.Image]:
        if img_bytes is None:
            return None
        img_cv = self._decode_cv(img_bytes)
        if img_cv is None:
            logger.warning("Gagal decode gambar, proses crop dilewati.")
            return None
        slot_dict = slot.model_dump()
        try:
            cropped_pil = self.yolo.crop_person(img_cv, int(slot_dict['w']), int(slot_dict['h']))
        finally:
            del img_cv
        if cropped_pil is None:
            logger.warning("Cropping gagal, mengembalikan None.")
        return cropped_pil

    async def _crop_photos_for_slots_parallel_async(self, images_bytes: List[Optional[bytes]], slots) -> List[Optional[Image.Image]]:
        num_slots = len(slots)
        valid = [b for b in images_bytes if b is not None]
        if not valid:
            logger.warning("Tidak ada gambar pengguna yang valid untuk di-crop.")
            return [None] * num_slots

        if len(valid) < num_slots:
            images_to_process = [valid[i % len(valid)] for i in range(num_slots)]
        else:
            images_to_process = valid[:num_slots]

        # threads (no ProcessPool → no pickling issues)
        tasks = [asyncio.to_thread(self._crop_one, b, s) for b, s in zip(images_to_process, slots)]
        return await asyncio.gather(*tasks)

    def _upload_static_page(self, index_str: str, frame_bytes: bytes, out_id: str, master_start_time: float) -> Tuple[int, str, float]:
        if frame_bytes is None:
            raise ValueError(f"Frame untuk halaman statis {index_str} tidak bisa dimuat.")
        pil_img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        public_id = f"{out_id}_page_{index_str}"
        start = time.perf_counter()
        url = upload_pil_image(pil_img, public_id=public_id, folder="images-scrapbook", fmt=SAVE_FORMAT, quality=JPEG_QUALITY)
        end = time.perf_counter()
        logger.info(f"Halaman Statis [{index_str}]: Upload selesai (T+{end - master_start_time:.2f}s).")
        pil_img.close()
        return (int(index_str), url, end - start)

    def _composite_one_page(self, index_str: str, frame_bytes: bytes, cropped_for_page: List[Optional[Image.Image]], slots, svg_paths, out_id: str, master_start_time: float) -> Tuple[int, str, float]:
        if frame_bytes is None:
            raise ValueError(f"Frame untuk halaman {index_str} tidak bisa dimuat.")
        frame_pil = Image.open(io.BytesIO(frame_bytes)).convert("RGBA")

        valid_crops = [im for im in cropped_for_page if im]
        if not valid_crops:
            logger.warning(f"Halaman [{index_str}]: Tidak ada foto valid, pembuatan halaman dilewati.")
            return (int(index_str), "", 0.0)

        slots_as_dicts = [s.model_dump() for s in slots]
        final_pil = image_process.insert_photos_with_svg_mask(frame_pil, cropped_for_page, slots_as_dicts, svg_paths)
        del frame_pil

        if final_pil is None:
            for im in cropped_for_page:
                if im: im.close()
            raise ValueError(f"Proses compositing untuk halaman {index_str} gagal.")

        public_id = f"{out_id}_page_{index_str}"
        start = time.perf_counter()
        url = upload_pil_image(final_pil.convert("RGB"), public_id=public_id, folder="images-scrapbook", fmt=SAVE_FORMAT, quality=JPEG_QUALITY)
        end = time.perf_counter()

        logger.info(f"Halaman [{index_str}]: Composite & Upload selesai (T+{end - master_start_time:.2f}s).")

        final_pil.close()
        for im in cropped_for_page:
            if im: im.close()
        return (int(index_str), url, end - start)

    async def process_template(self, template_data: TemplateData, request: Optional[Request] = None) -> Dict:
        run_id = template_data.id
        logger.info(f"=== START PROCESSING Run ID: {run_id} ===")
        try:
            process = psutil.Process(os.getpid())
            logger.info(f"Memory usage at start: {process.memory_info().rss / 1024 / 1024:.1f}MB for Run ID: {run_id}")
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")

        overall_start = time.perf_counter()
        try:
            results = {"run_id": run_id, "pages": [], "errors": []}
            page_keys_sorted = sorted(template_data.frame_images.keys(), key=int)

            pages_with_slots = {k: v for k, v in template_data.frame_images.items() if v.photo_slots}
            all_slots = [slot for k in sorted(pages_with_slots.keys()) for slot in pages_with_slots[k].photo_slots]

            # Stage 1: download
            logger.info(f"Tahap 1/4: Mengunduh {len(template_data.uploaded_images)} gambar pengguna dan {len(page_keys_sorted)} frame.")
            user_sources  = template_data.uploaded_images
            frame_sources = [template_data.frame_images[k].frame_image_path for k in page_keys_sorted]

            all_user_bytes, all_frame_bytes_list = await asyncio.gather(
                self._load_many_bytes_async(user_sources),
                self._load_many_bytes_async(frame_sources),
            )
            if request and await request.is_disconnected():
                logger.warning(f"[{run_id}] Client disconnected after downloads")
                raise asyncio.CancelledError()

            all_frame_bytes = dict(zip(page_keys_sorted, all_frame_bytes_list))
            logger.info(f"Tahap 1/4: Download selesai untuk Run ID: {run_id}")

            # Stage 2: cropping
            all_cropped: List[Optional[Image.Image]] = []
            if pages_with_slots:
                logger.info(f"Tahap 2/4: Melakukan smart-cropping untuk {len(all_slots)} slot.")
                crop_t0 = time.perf_counter()
                all_cropped = await self._crop_photos_for_slots_parallel_async(all_user_bytes, all_slots)
                logger.info(f"Tahap 2/4: Cropping selesai dalam {time.perf_counter()-crop_t0:.2f} detik untuk Run ID: {run_id}")
                if request and await request.is_disconnected():
                    logger.warning(f"[{run_id}] Client disconnected after cropping")
                    raise asyncio.CancelledError()
            else:
                logger.info(f"Tahap 2/4: Tidak ada slot foto, tahap cropping dilewati untuk Run ID: {run_id}")

            # Stage 3: composite + upload (threads)
            logger.info(f"Tahap 3/4: Memulai compositing dan upload paralel untuk Run ID: {run_id}")
            t_upload0 = time.perf_counter()
            tasks: List[asyncio.Task] = []

            cursor = 0
            for k in sorted(template_data.frame_images.keys(), key=int):
                page_cfg = template_data.frame_images[k]
                if page_cfg.photo_slots:
                    n = len(page_cfg.photo_slots)
                    photos = all_cropped[cursor: cursor + n]
                    cursor += n
                    task = asyncio.to_thread(
                        self._composite_one_page,
                        k,
                        all_frame_bytes.get(k),
                        photos,
                        page_cfg.photo_slots,
                        page_cfg.svg_paths,     # make sure this matches your schema name
                        run_id,
                        t_upload0,
                    )
                else:
                    task = asyncio.to_thread(
                        self._upload_static_page,
                        k,
                        all_frame_bytes.get(k),
                        run_id,
                        t_upload0,
                    )
                tasks.append(task)

            logger.info(f"Created {len(tasks)} thread tasks for Run ID: {run_id}")

            completed = 0
            failed    = 0
            async with asyncio.timeout(300):  # stage cap (independent of endpoint cap)
                for t in asyncio.as_completed(tasks):
                    if request and await request.is_disconnected():
                        logger.warning(f"[{run_id}] Client disconnected during uploads")
                        raise asyncio.CancelledError()
                    try:
                        page_idx, url, duration = await t
                        completed += 1
                        if url:
                            results["pages"].append({"index": page_idx, "url": url})
                            logger.info(f"Page {page_idx} OK ({completed}/{len(tasks)}) Run ID: {run_id}")
                        else:
                            logger.warning(f"Page {page_idx} finished but empty URL. Run ID: {run_id}")
                    except Exception as e:
                        failed += 1
                        logger.error(f"Task failed (Run ID {run_id}): {e}\n{traceback.format_exc()}")
                        results["errors"].append({"error": f"{type(e).__name__}: {e}"})

            logger.info(f"Futures completed: {completed}, failed: {failed}, total: {len(tasks)} for Run ID: {run_id}")
            logger.info(f"Tahap 3/4 selesai. Waktu proses: {time.perf_counter()-t_upload0:.2f}s untuk Run ID: {run_id}")

            # Stage 4: finalize
            logger.info(f"Tahap 4/4: Finalisasi hasil untuk Run ID: {run_id}")
            results["pages"].sort(key=lambda x: x["index"])

            logger.info(f"=== COMPLETED PROCESSING Run ID: {run_id} dalam {time.perf_counter()-overall_start:.2f} detik ===")

            # cleanup
            try:
                del all_cropped, all_frame_bytes, all_user_bytes, tasks, all_slots
                gc.collect()
            except Exception as cleanup_err:
                logger.warning(f"Cleanup error for Run ID {run_id}: {cleanup_err}")

            return results

        except asyncio.CancelledError:
            logger.warning(f"[{run_id}] Cancelled due to client disconnect or upstream timeout")
            raise
        except Exception as e:
            logger.error(f"=== CRITICAL ERROR in process_template for Run ID {run_id}: {e}\n{traceback.format_exc()} ===")
            raise
