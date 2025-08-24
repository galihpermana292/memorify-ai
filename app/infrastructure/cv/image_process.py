# app/infrastructure/cv/image_process.py
import cv2
from PIL import Image, ImageDraw
import numpy as np
from svgpathtools import parse_path
from ultralytics import YOLO

def sample_svg_path(svg_path_data, num_points=200):
    path = parse_path(svg_path_data)
    return [(p.real, p.imag) for p in [path.point(i / num_points) for i in range(num_points + 1)]]

def calculate_crop_coords(image_size, base_box, anchor_point, target_aspect_ratio):
    img_width, img_height = image_size
    box_w = base_box[2] - base_box[0]
    box_h = base_box[3] - base_box[1]
    box_area_ratio = (box_w * box_h) / (img_width * img_height)

    adaptive_padding = np.interp(box_area_ratio, [0.05, 0.5], [1.2, 1.05])

    crop_w_base = box_w * adaptive_padding
    crop_h_base = box_h * adaptive_padding
    current_aspect_ratio = crop_w_base / crop_h_base

    if current_aspect_ratio > target_aspect_ratio:
        new_h = crop_w_base / target_aspect_ratio
        new_w = crop_w_base
    else:
        new_w = crop_h_base * target_aspect_ratio
        new_h = crop_h_base

    anchor_x, anchor_y = anchor_point
    adjusted_anchor_y = anchor_y - new_h * 0.1

    crop_x1 = max(0, anchor_x - new_w / 2)
    crop_y1 = max(0, adjusted_anchor_y - new_h / 2)
    crop_x2 = min(img_width, crop_x1 + new_w)
    crop_y2 = min(img_height, crop_y1 + new_h)

    return (int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2))

def fallback_center_crop(image_pil, target_w, target_h):
    img_width, img_height = image_pil.size
    target_ratio = target_w / target_h
    img_ratio = img_width / img_height

    if img_ratio > target_ratio:
        new_width = int(target_ratio * img_height)
        left = (img_width - new_width) / 2
        top = 0
        right = left + new_width
        bottom = img_height
    else:
        new_height = int(img_width / target_ratio)
        left = 0
        top = (img_height - new_height) / 2
        right = img_width
        bottom = top + new_height

    cropped_img = image_pil.crop((left, top, right, bottom))
    return np.array(cropped_img)

def smart_crop_for_template(pose_model: YOLO, input_image_cv: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    if pose_model is None or input_image_cv is None:
        raise ValueError("Model atau gambar input tidak valid.")

    img_pil = Image.fromarray(cv2.cvtColor(input_image_cv, cv2.COLOR_BGR2RGB))
    
    # Panggilan ini sekarang akan menggunakan OpenVINO di backend secara transparan
    results = pose_model(img_pil, imgsz=640, verbose=False, classes=[0])
    result = results[0]

    # ... Sisa fungsi ini tidak perlu diubah sama sekali ...
    if len(result.boxes) == 0:
        return fallback_center_crop(img_pil, target_w, target_h)

    all_boxes = result.boxes.xyxy.cpu().numpy()
    min_x1, min_y1 = np.min(all_boxes[:, :2], axis=0)
    max_x2, max_y2 = np.max(all_boxes[:, 2:], axis=0)
    mega_box = np.array([min_x1, min_y1, max_x2, max_y2])
    mega_anchor_x, mega_anchor_y = (min_x1 + max_x2) / 2, (min_y1 + max_y2) / 2
    target_aspect_ratio = target_w / target_h

    crop_coords = calculate_crop_coords(img_pil.size, mega_box, (mega_anchor_x, mega_anchor_y), target_aspect_ratio)
    cropped_pil = img_pil.crop(crop_coords)
    
    return cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)

def crop_to_fill(image_pil: Image.Image, target_w: int, target_h: int) -> Image.Image:
    source_w, source_h = image_pil.size
    target_ratio = target_w / target_h
    source_ratio = source_w / source_h

    if source_ratio > target_ratio:
        scale_factor = target_h / source_h
        scaled_w = int(source_w * scale_factor)
        scaled_h = target_h
        resized_image = image_pil.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
        crop_x = (scaled_w - target_w) / 2
        return resized_image.crop((crop_x, 0, crop_x + target_w, scaled_h))
    else:
        scale_factor = target_w / source_w
        scaled_w = target_w
        scaled_h = int(source_h * scale_factor)
        resized_image = image_pil.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
        crop_y = (scaled_h - target_h) / 2
        return resized_image.crop((0, crop_y, scaled_w, crop_y + target_h))

def insert_photos_with_svg_mask(frame_pil: Image.Image, cropped_photos_pil: list, slots: list, svg_path_groups: list) -> Image.Image:
    final_frame = frame_pil.copy()

    for i, (photo_pil, slot) in enumerate(zip(cropped_photos_pil, slots)):
        paste_position = (int(slot["x"]), int(slot["y"]))
        target_w, target_h = int(slot["w"]), int(slot["h"])

        final_photo = crop_to_fill(photo_pil, target_w, target_h)

        if i < len(svg_path_groups) and svg_path_groups[i]:
            mask = Image.new("L", final_photo.size, 0)
            draw = ImageDraw.Draw(mask)
            for svg_path in svg_path_groups[i]:
                polygon = sample_svg_path(svg_path)
                draw.polygon(polygon, fill=255)
            final_frame.paste(final_photo, paste_position, mask=mask)
        else:
            final_frame.paste(final_photo, paste_position)

    return final_frame
