# app/infrastructure/cloudinary/upload_file.py
from io import BytesIO
from typing import Optional
from PIL import Image
import cloudinary, cloudinary.uploader
from app.config.settings import settings


# Configure once (supports CLOUDINARY_URL or split vars)
cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
    secure=True,
)

def upload_pil_image(
    img: Image.Image,
    public_id: str,
    folder: str = "images-scrapbook",
    fmt: str = "jpg",
    quality: int = 88,
    overwrite: bool = True,
    tags: Optional[list[str]] = None,
) -> str:
    fmt = (fmt or "jpg").lower()
    # Map to a valid Pillow format string
    if fmt in ("jpg", "jpeg"):
        pil_format = "JPEG"
        # JPEG can't have alpha
        if img.mode != "RGB":
            img = img.convert("RGB")
        save_kwargs = dict(format=pil_format, quality=quality, optimize=True)
    elif fmt == "png":
        pil_format = "PNG"
        # PNG supports alpha; keep mode as-is
        save_kwargs = dict(format=pil_format, optimize=True)
    else:
        pil_format = fmt.upper()
        save_kwargs = dict(format=pil_format)

    buf = BytesIO()
    img.save(buf, **save_kwargs)
    buf.seek(0)

    res = cloudinary.uploader.upload(
        buf,
        resource_type="image",
        folder=folder,
        public_id=public_id,
        overwrite=overwrite,
        format=fmt,              # final extension in Cloudinary
        tags=tags or [],
    )
    return res["secure_url"]
