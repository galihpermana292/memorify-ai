from fastapi import APIRouter, Request # <-- Add Request
from fastapi.concurrency import run_in_threadpool
from app.delivery.schemas.body import TemplateData

router = APIRouter()

@router.post("/process-template")
async def process_template(request: Request, template_data: TemplateData):
    # --- ACTION: Get the pre-loaded service from app.state ---
    service = request.app.state.template_service
    
    # --- ACTION: Run the blocking CV code in a thread pool ---
    # This is the most critical step for performance.
    # We call a synchronous worker function to do the heavy lifting.
    result_paths = await run_in_threadpool(service.process_template, template_data)
    
    return {"output_files": result_paths}