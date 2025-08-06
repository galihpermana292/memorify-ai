import uuid
from sqlalchemy import Column, String, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Template(Base):
    __tablename__ = "template"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    label = Column(String)
    type = Column(String)
    tag = Column(JSON)  # or ARRAY(String) if you want
    slug = Column(String)
    category = Column(String)
    thumbnail_uri = Column(String)
    create_time = Column(DateTime)
    update_time = Column(DateTime)
    frame_data = Column(JSON)  # This holds frame_image_path, svg_paths, photo_slots
