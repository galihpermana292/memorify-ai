# app/delivery/api/computer_vision.py
from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from app.delivery.schemas.body import TemplateData
from app.config.settings import settings
import secrets
import logging

router = APIRouter()
security = HTTPBasic()
logger = logging.getLogger("uvicorn.access")

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
    service = request.app.state.template_service
    result = await service.process_template(template_data)

    return {"output_files": result}


@router.get("/warm")
async def warm(request: Request):
 
    # triggers lazy init via middleware
    svc = getattr(request.app.state, "template_service", None)
    if svc is None:
        # if middleware hasn’t run yet, force init here:
        from app.domain.yolo_processor import YOLOProcessor
        from app.domain.template_service import TemplateService
        yp = YOLOProcessor()
        request.app.state.template_service = TemplateService(yolo=yp)
        svc = request.app.state.template_service

    try:
        # If you added YOLOProcessor.warmup():
        svc.yolo_processor.warmup(imgsz=320)
        warmed = True
    except Exception:
        # fallback warm via a direct predict call
        try:
            from PIL import Image
            _ = svc.yolo_processor.model.predict(Image.new("RGB", (320, 320)), imgsz=320, verbose=False, device="cpu")
            warmed = True
        except Exception:
            warmed = False

    return {"warmed": warmed}