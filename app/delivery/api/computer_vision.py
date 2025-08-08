from fastapi import APIRouter
from app.domain.yolo_processor import YOLOProcessor
from app.domain.template_service import TemplateService
from app.delivery.schemas.body import TemplateData

router = APIRouter()

@router.post("/process-template")
async def process_template(template_data: TemplateData):
    yolo = YOLOProcessor()
    service = TemplateService(yolo)
    
    result_paths = await service.process_template(template_data)
    return {"output_files": result_paths}
