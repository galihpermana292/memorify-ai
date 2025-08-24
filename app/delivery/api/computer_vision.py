# app/delivery/api/computer_vision.py
from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from app.delivery.schemas.body import TemplateData
from app.config.settings import settings
import secrets
import logging
import traceback

router = APIRouter()
security = HTTPBasic()
# Mengambil logger yang sudah ada yang dikonfigurasi oleh Uvicorn
logger = logging.getLogger("uvicorn.error")

def verify_basic_auth(creds: HTTPBasicCredentials = Depends(security)) -> None:
    ok_user = secrets.compare_digest(creds.username, settings.BASIC_AUTH_USERNAME)
    ok_pass = secrets.compare_digest(creds.password, settings.BASIC_AUTH_PASSWORD)
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

@router.post("/process-template", dependencies=[Depends(verify_basic_auth)])
async def process_template(request: Request, template_data: TemplateData):
    try:
        service = request.app.state.template_service
        result = await service.process_template(template_data)
        return {"output_files": result}
    except Exception as e:
        logger.error(f"Error tidak tertangani di endpoint process-template: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Terjadi kesalahan internal pada server."
        )

@router.get("/warm")
async def warm(request: Request):
    svc = getattr(request.app.state, "template_service", None)
    # Logika _ensure_service di main.py akan menangani inisialisasi jika svc adalah None
    if svc is None:
        return { "status": "error", "message": "Service is not initialized." }

    warmed = False
    try:
        svc.yolo_processor.warmup(imgsz=320)
        warmed = True
    except Exception as e:
        logger.error(f"Pemanasan model gagal: {e}", exc_info=True)
        warmed = False

    return {
        "status": "ok" if warmed else "error",
        "model_loaded": warmed,
        "message": "Model YOLO siap digunakan." if warmed else "Gagal melakukan pemanasan model."
    }