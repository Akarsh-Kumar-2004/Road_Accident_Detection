from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


PLATE_REGEX = re.compile(r"^[A-Z0-9]{7,11}$")


@dataclass
class PlateCandidate:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    image: np.ndarray


def detect_number_plate(frame: np.ndarray, vehicle_regions: Sequence[Tuple[int, int, int, int]], reader) -> Optional[PlateCandidate]:
    candidates: List[PlateCandidate] = []

    for vehicle_bbox in vehicle_regions:
        vehicle_crop, vehicle_origin = _crop_with_padding(frame, vehicle_bbox, x_pad=0.08, y_pad=0.12)
        if vehicle_crop.size == 0:
            continue

        for local_bbox, plate_crop in _generate_plate_regions(vehicle_crop):
            candidate = _run_ocr_on_plate_crop(
                plate_crop=plate_crop,
                reader=reader,
                offset=(vehicle_origin[0] + local_bbox[0], vehicle_origin[1] + local_bbox[1]),
            )
            if candidate is not None:
                candidates.append(candidate)

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item.confidence, len(item.text)), reverse=True)
    return candidates[0]


def save_plate_image(candidate: PlateCandidate, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, candidate.image)
    return output_path


def _generate_plate_regions(vehicle_crop: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    grayscale = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(grayscale, 11, 17, 17)
    edges = cv2.Canny(filtered, 30, 180)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    regions: List[Tuple[Tuple[int, int, int, int], np.ndarray]] = []
    seen_boxes = set()
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 600:
            continue

        aspect_ratio = w / max(h, 1)
        if not 2.0 <= aspect_ratio <= 6.5:
            continue

        relative_width = w / max(vehicle_crop.shape[1], 1)
        relative_height = h / max(vehicle_crop.shape[0], 1)
        if not 0.12 <= relative_width <= 0.9:
            continue
        if not 0.05 <= relative_height <= 0.35:
            continue

        box = _expand_box(x, y, w, h, vehicle_crop.shape[1], vehicle_crop.shape[0], pad_x=0.08, pad_y=0.18)
        if box in seen_boxes:
            continue
        seen_boxes.add(box)

        x1, y1, x2, y2 = box
        plate_crop = vehicle_crop[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue
        regions.append((box, plate_crop))

    if not regions:
        h, w = vehicle_crop.shape[:2]
        fallback_box = (
            int(w * 0.20),
            int(h * 0.45),
            int(w * 0.80),
            int(h * 0.80),
        )
        x1, y1, x2, y2 = fallback_box
        fallback_crop = vehicle_crop[y1:y2, x1:x2]
        if fallback_crop.size != 0:
            regions.append((fallback_box, fallback_crop))

    return regions


def _run_ocr_on_plate_crop(
    plate_crop: np.ndarray,
    reader,
    offset: Tuple[int, int],
) -> Optional[PlateCandidate]:
    best_candidate: Optional[PlateCandidate] = None

    for processed in _plate_variants(plate_crop):
        results = reader.readtext(processed, detail=1, paragraph=False, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        for bbox, text, confidence in results:
            cleaned = _normalize_plate_text(text)
            if not _looks_like_plate(cleaned):
                continue

            x_coords = [int(point[0]) for point in bbox]
            y_coords = [int(point[1]) for point in bbox]
            local_box = (
                max(min(x_coords), 0),
                max(min(y_coords), 0),
                min(max(x_coords), plate_crop.shape[1]),
                min(max(y_coords), plate_crop.shape[0]),
            )
            if local_box[2] <= local_box[0] or local_box[3] <= local_box[1]:
                local_box = (0, 0, plate_crop.shape[1], plate_crop.shape[0])

            crop = plate_crop[local_box[1]:local_box[3], local_box[0]:local_box[2]]
            absolute_box = (
                offset[0] + local_box[0],
                offset[1] + local_box[1],
                offset[0] + local_box[2],
                offset[1] + local_box[3],
            )
            candidate = PlateCandidate(
                text=cleaned,
                confidence=float(confidence),
                bbox=absolute_box,
                image=crop if crop.size != 0 else plate_crop,
            )
            if best_candidate is None or (candidate.confidence, len(candidate.text)) > (
                best_candidate.confidence,
                len(best_candidate.text),
            ):
                best_candidate = candidate

    return best_candidate


def _plate_variants(plate_crop: np.ndarray) -> List[np.ndarray]:
    grayscale = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(grayscale, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    equalized = cv2.equalizeHist(enlarged)
    _, threshold = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        equalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return [enlarged, equalized, threshold, adaptive]


def _crop_with_padding(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    *,
    x_pad: float,
    y_pad: float,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    pad_x = int(width * x_pad)
    pad_y = int(height * y_pad)

    crop_x1 = max(x1 - pad_x, 0)
    crop_y1 = max(y1 - pad_y, 0)
    crop_x2 = min(x2 + pad_x, frame.shape[1])
    crop_y2 = min(y2 + pad_y, frame.shape[0])
    return frame[crop_y1:crop_y2, crop_x1:crop_x2], (crop_x1, crop_y1)


def _expand_box(
    x: int,
    y: int,
    w: int,
    h: int,
    max_width: int,
    max_height: int,
    *,
    pad_x: float,
    pad_y: float,
) -> Tuple[int, int, int, int]:
    extra_x = int(w * pad_x)
    extra_y = int(h * pad_y)
    x1 = max(x - extra_x, 0)
    y1 = max(y - extra_y, 0)
    x2 = min(x + w + extra_x, max_width)
    y2 = min(y + h + extra_y, max_height)
    return x1, y1, x2, y2


def _normalize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _looks_like_plate(text: str) -> bool:
    if not PLATE_REGEX.match(text):
        return False

    alpha_count = sum(character.isalpha() for character in text)
    digit_count = sum(character.isdigit() for character in text)
    return alpha_count >= 2 and digit_count >= 2
