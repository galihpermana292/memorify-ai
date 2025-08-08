from pydantic import BaseModel
from typing import List, Optional, Dict

class TemplateData(BaseModel):
    id: str
    name: str
    label: str
    type: str
    tag: List[str]
    slug: str
    category: str
    thumbnail_uri: Optional[str]
    frame_images: Dict[str, Dict[str, str]]
    svg_files: List[str]
    photo_slots: List[Dict[str, int]]
    uploaded_images: List[str]