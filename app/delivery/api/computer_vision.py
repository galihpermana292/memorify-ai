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
    request_id = template_data.id
    thread_count = threading.active_count()
    logger.info(f"=== ENDPOINT START for {request_id} (Active threads: {thread_count}) ===")
    
    try:
        # Verify service is available
        if not hasattr(request.app.state, 'template_service'):
            logger.error(f"Service not initialized for request {request_id}")
            raise HTTPException(
                status_code=500, 
                detail="Service not available"
            )
        
        if not hasattr(request.app.state, 'executor'):
            logger.error(f"Executor not initialized for request {request_id}")
            raise HTTPException(
                status_code=500, 
                detail="Executor not available"
            )
        
        service = request.app.state.template_service
        logger.info(f"Service and executor retrieved for {request_id}")
        
        # Add timeout to the entire operation
        try:
            result = await asyncio.wait_for(
                service.process_template(template_data),
                timeout=600  # 10 minute timeout
            )
            logger.info(f"=== ENDPOINT SUCCESS for {request_id} ===")
            
            # Explicitly return JSONResponse
            return JSONResponse(
                status_code=200,
                content={"output_files": result}
            )
            
        except asyncio.TimeoutError:
            logger.error(f"=== ENDPOINT TIMEOUT for {request_id} ===")
            raise HTTPException(
                status_code=408,
                detail="Request timeout"
            )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"=== ENDPOINT ERROR for {request_id}: {e} ===\n{traceback.format_exc()}")
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
        svc.yolo_processor.warmup(imgsz=416)
        warmed = True
    except Exception as e:
        logger.error(f"Pemanasan model gagal: {e}", exc_info=True)
        warmed = False

    return {
        "status": "ok" if warmed else "error",
        "model_loaded": warmed,
        "message": "Model YOLO siap digunakan." if warmed else "Gagal melakukan pemanasan model."
    }