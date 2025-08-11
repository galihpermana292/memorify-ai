# config/settings.py
from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # API
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ML Service"

    # CORS
    ALLOWED_HOSTS: List[str] = ["*"]

    # ML
    YOLO_MODEL_PATH: str = "models/yolov10x.pt"

    # Auth
    BASIC_AUTH_USERNAME: str = "admin"
    BASIC_AUTH_PASSWORD: str = "secret"

    # Templates
    TEMPLATES_DIR: str = "templates"

    # External APIs
    GOLANG_API_URL: str = "http://localhost:8080"
    GOLANG_API_KEY: Optional[str] = None

    # Env
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Cloudinary (either use CLOUDINARY_URL or the 3 fields below)
    CLOUDINARY_URL: Optional[str] = "cloudinary://322937667181681:oXA4pG_QEklWRCrPYOvUMlrf2WM@dduonada5"
    CLOUDINARY_CLOUD_NAME: Optional[str] = None
    CLOUDINARY_API_KEY: Optional[str] = None
    CLOUDINARY_API_SECRET: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
