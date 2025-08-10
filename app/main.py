from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.config.settings import settings
from app.delivery.api.computer_vision import router
from app.domain.yolo_processor import YOLOProcessor
from app.domain.template_service import TemplateService


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting ML Service...")
    print("Loading YOLO model...")
    
    yolo_processor = YOLOProcessor() # This will trigger the download/load
    app.state.template_service = TemplateService(yolo=yolo_processor)
    print("✅ Model loaded and service is ready.")
    
    yield
    
    print("🛑 Shutting down ML Service...")


app = FastAPI(
    title="ML Computer Vision Service",
    description="Machine Learning service for computer vision tasks including frame detection, smart cropping, and photo insertion",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    return {
        "message": "ML Computer Vision Service",
        "version": "1.0.0",
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ml-computer-vision"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
