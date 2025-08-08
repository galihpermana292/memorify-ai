from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ML Service"

    # CORS
    ALLOWED_HOSTS: List[str] = ["*"]

    # ML Models
    YOLO_MODEL_PATH: str = "models/yolov10x.pt"

    # Templates
    TEMPLATES_DIR: str = "templates"

    # External APIs
    GOLANG_API_URL: str = "http://localhost:8080"
    GOLANG_API_KEY: Optional[str] = None

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
