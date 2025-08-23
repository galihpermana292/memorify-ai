from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import threading

from app.config.settings import settings
from app.delivery.api.computer_vision import router
from app.domain.template_service import TemplateService
from app.domain.yolo_processor import YOLOProcessor

# --- Lazy service bootstrap state ---
_service_lock = threading.Lock()
_service_ready = False

def _ensure_service(app: FastAPI) -> None:
    """
    Initialize YOLO + TemplateService exactly once, on-demand.
    Safe to call concurrently; only the first caller does real work.
    """
    global _service_ready
    if _service_ready:
        return
    with _service_lock:
        if _service_ready:
            return
        yolo_processor = YOLOProcessor()
        app.state.template_service = TemplateService(yolo=yolo_processor)
        _service_ready = True
        print("✅ YOLO + TemplateService initialized (lazy).")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting ML Service (lazy init enabled)…")
    yield
    print("🛑 Shutting down ML Service…")

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
    if request.url.path.startswith(settings.API_V1_STR):
        _ensure_service(request.app)
    return await call_next(request)

# Include your CV router
app.include_router(router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {"message": "ML Computer Vision Service", "version": "1.0.0", "status": "ok"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "ml-computer-vision", "model_loaded": _service_ready}

@app.get("/health/cloudinary")
def cloudinary_health():
    if not (settings.CLOUDINARY_URL or (
        settings.CLOUDINARY_CLOUD_NAME and settings.CLOUDINARY_API_KEY and settings.CLOUDINARY_API_SECRET
    )):
        raise HTTPException(500, "Cloudinary credentials not configured")
    return {"status": "ok"}
