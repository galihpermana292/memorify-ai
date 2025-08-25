# app/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
import os

from app.config.settings import settings
from app.delivery.api.computer_vision import router
from app.domain.template_service import TemplateService
from app.domain.yolo_processor import YOLOProcessor

logging.getLogger("ultralytics").setLevel(logging.WARNING)
logger = logging.getLogger("uvicorn.error")

# --- Lazy service bootstrap state ---
_service_lock = threading.Lock()
_service_ready = False

def _ensure_service(app: FastAPI) -> None:
    global _service_ready
    with _service_lock:  # Always acquire lock first
        if _service_ready:
            return
        logger.info("Memulai inisialisasi TemplateService dan YOLOProcessor (lazy-init)...")
        yolo_processor = YOLOProcessor()
        app.state.template_service = TemplateService(
            yolo=yolo_processor,
            executor=app.state.executor
        )
        _service_ready = True
        logger.info("Inisialisasi service selesai.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    max_workers = min(4, os.cpu_count() or 1)  # Conservative limit
    app.state.executor = ThreadPoolExecutor(max_workers=max_workers)
    logger.info(f"ML Service '{settings.PROJECT_NAME}' dimulai (mode: {settings.ENVIRONMENT}).")
    logger.info(f"Shared ThreadPoolExecutor dibuat dengan {max_workers} workers.")
    yield
    logger.info("Menutup ThreadPoolExecutor...")
    app.state.executor.shutdown(wait=True)
    logger.info("ML Service berhenti.")

app = FastAPI(
    title="ML Computer Vision Service",
    description="Machine Learning service for computer vision tasks including frame detection, smart cropping, and photo insertion",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load only for API routes
@app.middleware("http")
async def lazy_boot(request: Request, call_next):
    # Inisialisasi service hanya jika path request berada di bawah API_V1_STR
    if request.url.path.startswith(settings.API_V1_STR):
        _ensure_service(request.app)
    return await call_next(request)

# Include your CV router
app.include_router(router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {"message": "Memo AI Computer Vision Service", "version": "1.0.0", "status": "ok"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Memo AI 1.0", "model_loaded": _service_ready}