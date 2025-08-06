import cv2
from PIL import Image, ImageDraw
import numpy as np
import os
import time
from svgpathtools import parse_path
import random

# ====== Utility: Sample SVG path into polygon points ======
def sample_svg_path(svg_path_data, num_points=200):
    path = parse_path(svg_path_data)
    return [(p.real, p.imag) for p in [path.point(i / num_points) for i in range(num_points + 1)]]

# ====== Utility: Group and merge detected person boxes ======
def group_and_merge_boxes(person_boxes, threshold_factor=1.5):
    if not person_boxes:
        return []

    num_boxes = len(person_boxes)
    adj = [[] for _ in range(num_boxes)]

    for i in range(num_boxes):
        for j in range(i + 1, num_boxes):
            box1, box2 = person_boxes[i], person_boxes[j]
            center1_x, center1_y = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
            center2_x, center2_y = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
            width1, width2 = box1[2] - box1[0], box2[2] - box2[0]
            distance = np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)
            distance_threshold = threshold_factor * (width1 + width2) / 2
            if distance < distance_threshold:
                adj[i].append(j)
                adj[j].append(i)

    visited, groups = [False] * num_boxes, []
    for i in range(num_boxes):
        if not visited[i]:
            current_group_indices, stack = [], [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                current_group_indices.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            groups.append(current_group_indices)

    merged_boxes = []
    for group_indices in groups:
        min_x1, min_y1 = float('inf'), float('inf')
        max_x2, max_y2 = float('-inf'), float('-inf')
        for index in group_indices:
            box = person_boxes[index]
            min_x1 = min(min_x1, box[0])
            min_y1 = min(min_y1, box[1])
            max_x2 = max(max_x2, box[2])
            max_y2 = max(max_y2, box[3])
        merged_boxes.append([min_x1, min_y1, max_x2, max_y2])
    return merged_boxes

# ====== Smart crop using YOLO detection ======
def smart_crop_with_yolo(model, input_image_cv, target_w=375, target_h=375, fallback_threshold=0.30):
    if model is None or input_image_cv is None:
        return None

    img_rgb = cv2.cvtColor(input_image_cv, cv2.COLOR_RGBA2RGB)
    results = model(img_rgb, verbose=False)

    person_boxes = [box.xyxy[0].cpu().numpy() for box in results[0].boxes if model.names[int(box.cls[0])] == 'person']

    if not person_boxes:
        return pad_to_target_aspect(img_rgb, target_w, target_h)

    merged_boxes = group_and_merge_boxes(person_boxes)
    largest_box = max(merged_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x1, y1, x2, y2 = map(int, largest_box)

    cropped_image_rgb = img_rgb[y1:y2, x1:x2]
    if cropped_image_rgb.size == 0:
        return pad_to_target_aspect(img_rgb, target_w, target_h)

    actual_crop_h, actual_crop_w = cropped_image_rgb.shape[:2]
    actual_crop_ratio = actual_crop_w / actual_crop_h
    target_ratio = target_w / target_h

    deviation = abs(actual_crop_ratio - target_ratio)
    if deviation > fallback_threshold:
        return fallback_center_crop(input_image_cv, target_w, target_h)

    return pad_to_target_aspect(cropped_image_rgb, target_w, target_h, base_image=img_rgb)

# ====== Fallback center crop ======
def fallback_center_crop(image_rgba, target_w, target_h):
    img_rgb = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2RGB)
    img_height, img_width, _ = img_rgb.shape
    target_ratio = target_w / target_h

    center_x, center_y = img_width // 2, img_height // 2

    if img_width / img_height >= target_ratio:
        crop_h = img_height
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = img_width
        crop_h = int(crop_w / target_ratio)

    x1 = max(0, center_x - crop_w // 2)
    y1 = max(0, center_y - crop_h // 2)
    x2 = min(x1 + crop_w, img_width)
    y2 = min(y1 + crop_h, img_height)

    cropped = img_rgb[y1:y2, x1:x2]
    return pad_to_target_aspect(cropped, target_w, target_h, base_image=img_rgb)

# ====== Pad image to target aspect ratio ======
def pad_to_target_aspect(image_rgb, target_w, target_h, base_image=None):
    target_ratio = target_w / target_h
    h, w = image_rgb.shape[:2]
    current_ratio = w / h

    if abs(current_ratio - target_ratio) < 1e-2:
        return image_rgb

    if current_ratio > target_ratio:
        new_w = w
        new_h = int(w / target_ratio)
    else:
        new_h = h
        new_w = int(h * target_ratio)

    pad_top = (new_h - h) // 2
    pad_bottom = new_h - h - pad_top
    pad_left = (new_w - w) // 2
    pad_right = new_w - w - pad_left

    if base_image is not None:
        try:
            replicated = cv2.copyMakeBorder(
                image_rgb, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_REPLICATE
            )
            return replicated
        except:
            pass

    padded = cv2.copyMakeBorder(
        image_rgb, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)
    )
    return padded

# ====== Insert photos into frame using SVG mask ======
def insert_photos_with_svg_mask(frame_path, cropped_photos_pil, slots, svg_path_groups):
    try:
        frame = Image.open(frame_path).convert("RGBA")
    except FileNotFoundError:
        return None

    for i, (photo_pil, slot) in enumerate(zip(cropped_photos_pil, slots)):
        target_w = int(slot["w"])
        target_h = int(slot["h"])
        paste_x = int(slot["x"])
        paste_y = int(slot["y"])

        resized_photo = photo_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)

        if i < len(svg_path_groups) and svg_path_groups[i]:
            combined_mask = Image.new("L", (target_w, target_h), 0)
            draw = ImageDraw.Draw(combined_mask)
            for svg_path_data in svg_path_groups[i]:
                polygon = sample_svg_path(svg_path_data)
                draw.polygon(polygon, fill=255)

            masked_photo = Image.new("RGBA", (target_w, target_h))
            masked_photo.paste(resized_photo, (0, 0), mask=combined_mask)
            frame.paste(masked_photo, (paste_x, paste_y), mask=masked_photo.split()[3])
        else:
            frame.paste(resized_photo, (paste_x, paste_y))

    return frame
