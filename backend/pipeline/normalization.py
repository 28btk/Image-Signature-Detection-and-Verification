from __future__ import annotations

import cv2
import numpy as np

from backend.config import Config
from backend.pipeline.preprocessing import preprocess_signature_image


def find_foreground_bbox(mask: np.ndarray, min_pixels: int = 25) -> tuple[int, int, int, int]:
    points = np.column_stack(np.where(mask > 0))
    if len(points) < min_pixels:
        raise ValueError("The signature image does not contain enough visible ink pixels.")

    y_values = points[:, 0]
    x_values = points[:, 1]

    x_min, x_max = int(x_values.min()), int(x_values.max())
    y_min, y_max = int(y_values.min()), int(y_values.max())
    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def crop_mask(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return mask[y : y + h, x : x + w].copy()


def normalize_signature(
    image: np.ndarray,
    target_size: tuple[int, int] | None = None,
    padding: int = 12,
) -> dict[str, object]:
    target_size = target_size or (Config.NORMALIZED_WIDTH, Config.NORMALIZED_HEIGHT)
    target_width, target_height = target_size

    processed = preprocess_signature_image(image)
    ink_mask = processed["binary"]
    bbox = find_foreground_bbox(ink_mask)
    cropped_mask = crop_mask(ink_mask, bbox)

    usable_width = max(target_width - (padding * 2), 1)
    usable_height = max(target_height - (padding * 2), 1)
    scale = min(
        usable_width / cropped_mask.shape[1],
        usable_height / cropped_mask.shape[0],
    )

    resized_width = max(int(round(cropped_mask.shape[1] * scale)), 1)
    resized_height = max(int(round(cropped_mask.shape[0] * scale)), 1)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized_mask = cv2.resize(cropped_mask, (resized_width, resized_height), interpolation=interpolation)

    normalized_image = np.full((target_height, target_width), 255, dtype=np.uint8)
    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2

    normalized_image[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = 255 - resized_mask
    normalized_ink_mask = 255 - normalized_image

    return {
        "normalized_image": normalized_image,
        "ink_mask": normalized_ink_mask,
        "source_bbox": bbox,
        "target_size": {
            "width": target_width,
            "height": target_height,
        },
    }
