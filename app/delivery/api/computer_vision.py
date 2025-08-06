from fastapi import APIRouter, Depends, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from app.config.database import get_db
from app.domain.yolo_processor import YOLOProcessor
from app.domain.template_service import TemplateService

router = APIRouter()

@router.post("/process-template/{template_id}")
async def process_template(template_id: str, files: list[UploadFile], db: AsyncSession = Depends(get_db)):
    yolo = YOLOProcessor()
    service = TemplateService(db, yolo)
    images = [await f.read() for f in files]
    result_paths = await service.process_template(template_id, images)
    return {"output_files": result_paths}
