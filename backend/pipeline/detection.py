from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.config import Config
from backend.pipeline.preprocessing import preprocess_document_image


@dataclass
class DetectionCandidate:
    bbox: tuple[int, int, int, int]
    score: float
    contour_area: float
    fill_ratio: float


def _density_score(fill_ratio: float) -> float:
    target = 0.26
    distance = abs(fill_ratio - target)
    return max(0.0, 1.0 - (distance / target))


def _score_candidate(contour: np.ndarray, image_shape: tuple[int, int, int]) -> DetectionCandidate | None:
    image_height, image_width = image_shape[:2]
    x, y, w, h = cv2.boundingRect(contour)
    contour_area = cv2.contourArea(contour)

    if contour_area < image_width * image_height * Config.DETECTION_MIN_AREA_RATIO:
        return None

    width_ratio = w / image_width
    height_ratio = h / image_height
    aspect_ratio = w / max(h, 1)
    fill_ratio = contour_area / max(w * h, 1)
    vertical_bias = (y + (h * 0.5)) / image_height

    if width_ratio < 0.08 or height_ratio > 0.55:
        return None

    aspect_score = max(0.0, 1.0 - abs(aspect_ratio - 4.0) / 4.0)
    score = (
        (1.9 * width_ratio)
        + (1.1 * vertical_bias)
        + (0.7 * aspect_score)
        + (0.5 * _density_score(fill_ratio))
    )

    return DetectionCandidate(
        bbox=(x, y, w, h),
        score=float(score),
        contour_area=float(contour_area),
        fill_ratio=float(fill_ratio),
    )


def expand_bbox(bbox: tuple[int, int, int, int], image_shape: tuple[int, int, int], padding_ratio: float = 0.08):
    image_height, image_width = image_shape[:2]
    x, y, w, h = bbox

    pad_x = max(int(w * padding_ratio), 10)
    pad_y = max(int(h * padding_ratio), 8)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image_width, x + w + pad_x)
    y2 = min(image_height, y + h + pad_y)
    return (x1, y1, x2 - x1, y2 - y1)


def crop_bbox(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return image[y : y + h, x : x + w].copy()


def draw_bbox(image: np.ndarray, bbox: tuple[int, int, int, int], label: str = "Signature") -> np.ndarray:
    x, y, w, h = bbox
    annotated = image.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 140, 255), 2)
    cv2.putText(
        annotated,
        label,
        (x, max(25, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 140, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def detect_signature_region(image: np.ndarray) -> dict[str, object]:
    processed = preprocess_document_image(image)
    mask = processed["mask"]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No signature-like region could be found in the uploaded image.")

    candidates: list[DetectionCandidate] = []
    for contour in contours:
        candidate = _score_candidate(contour, image.shape)
        if candidate is not None:
            candidates.append(candidate)

    if candidates:
        best_candidate = max(candidates, key=lambda item: item.score)
    else:
        fallback_contour = max(contours, key=cv2.contourArea)
        fallback_bbox = cv2.boundingRect(fallback_contour)
        contour_area = cv2.contourArea(fallback_contour)
        fill_ratio = contour_area / max(fallback_bbox[2] * fallback_bbox[3], 1)
        best_candidate = DetectionCandidate(
            bbox=fallback_bbox,
            score=0.0,
            contour_area=float(contour_area),
            fill_ratio=float(fill_ratio),
        )

    final_bbox = expand_bbox(best_candidate.bbox, image.shape)
    annotated_image = draw_bbox(image, final_bbox)
    cropped_signature = crop_bbox(image, final_bbox)

    return {
        "bbox": final_bbox,
        "candidate_score": round(best_candidate.score, 4),
        "contour_area": round(best_candidate.contour_area, 2),
        "fill_ratio": round(best_candidate.fill_ratio, 4),
        "candidate_count": len(candidates),
        "mask": mask,
        "annotated_image": annotated_image,
        "cropped_signature": cropped_signature,
    }
