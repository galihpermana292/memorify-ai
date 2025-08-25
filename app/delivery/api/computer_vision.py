# app/delivery/api/computer_vision.py
from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from app.delivery.schemas.body import TemplateData
from app.config.settings import settings
import secrets
import threading
import logging
import traceback
import asyncio

router = APIRouter()
security = HTTPBasic()
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

ENDPOINT_TIMEOUT_SECONDS = 55

@router.post("/process-template", dependencies=[Depends(verify_basic_auth)])
async def process_template(request: Request, template_data: TemplateData):
    request_id = template_data.id
    logger.info(f"=== ENDPOINT START for {request_id} (threads={threading.active_count()}) ===")

    try:
        service = getattr(request.app.state, "template_service", None)
        if service is None:
            logger.error(f"Service not initialized for request {request_id}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service is not ready. Please try again in a moment.",
            )

        # Abort fast if the client already closed
        if await request.is_disconnected():
            logger.warning(f"[{request_id}] Client already disconnected")
            raise HTTPException(status_code=499, detail="Client closed request")

        try:
            result = await asyncio.wait_for(
                service.process_template(template_data, request=request),
                timeout=ENDPOINT_TIMEOUT_SECONDS,
            )
            logger.info(f"=== ENDPOINT SUCCESS for {request_id} ===")
            return JSONResponse(status_code=200, content={"output_files": result})

        except asyncio.TimeoutError:
            logger.error(f"=== ENDPOINT TIMEOUT for {request_id} after {ENDPOINT_TIMEOUT_SECONDS}s ===")
            raise HTTPException(status_code=504, detail="AI processing timed out")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"=== ENDPOINT ERROR for {request_id}: {e} ===\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Terjadi kesalahan internal pada server.",
        )

@router.get("/warm")
async def warm(request: Request):
    svc = getattr(request.app.state, "template_service", None)
    if svc is None:
        return {"status": "error", "message": "Service is not initialized."}

    try:
        await svc.warmup()   # call the service-level warmup
        return {"status": "ok", "model_loaded": True, "message": "Model YOLO siap digunakan."}
    except Exception as e:
        logger.error(f"Pemanasan model gagal: {e}", exc_info=True)
        return {"status": "error", "model_loaded": False, "message": "Gagal melakukan pemanasan model."}
