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

# --- Application lifespan for resource management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    cpu_workers = min(os.cpu_count() or 1, 4)
    io_workers = min(cpu_workers * 2, 8)

    app.state.cpu_executor = ThreadPoolExecutor(max_workers=cpu_workers)
    app.state.io_executor = ThreadPoolExecutor(max_workers=io_workers)
    
    logger.info("Memulai inisialisasi TemplateService dan YOLOProcessor...")
    yolo_processor = YOLOProcessor()
    
    app.state.template_service = TemplateService(
        yolo=yolo_processor,
        cpu_executor=app.state.cpu_executor,
        io_executor=app.state.io_executor
    )
    logger.info("Inisialisasi service selesai.")
    
    yield
    
    logger.info("Menutup ThreadPoolExecutors...")
    app.state.cpu_executor.shutdown(wait=True)
    app.state.io_executor.shutdown(wait=True)
    logger.info("ML Service berhenti.")

app = FastAPI(
    title="Memo AI 1.0 Computer Vision Service",
    description="Machine Learning service for computer vision tasks including frame detection, smart cropping, and photo insertion",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
)

# --- Middleware and API Endpoints ---

# CORS setup for handling cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your API routes are included here
app.include_router(router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {"message": "Memo AI Computer Vision Service", "version": "1.0.0", "status": "ok"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Memo AI 1.0", "model_loaded": True}