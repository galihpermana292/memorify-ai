from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.config.settings import settings
from app.config.database import init_db
from app.delivery.api.computer_vision import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting ML Service...")
    await init_db()
    yield
    # Shutdown
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
