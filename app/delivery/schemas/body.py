from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class Slot(BaseModel):
    x: float
    y: float
    w: float
    h: float

class PageData(BaseModel):
    frame_image_path: str                  # URL or path
    photo_slots: List[Slot] = Field(default_factory=list)
    svg_paths: List[List[str]] = Field(default_factory=list)  # groups of SVG path strings

class TemplateData(BaseModel):
    id: str
    name: str
    type: str

    # Per-page data keyed by page index: "0","1","2"... (JSON keys are strings)
    frame_images: Dict[str, PageData]

    # User photos to place into slots; URLs or base64 (data URLs supported)
    uploaded_images: List[str]