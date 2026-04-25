from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from backend.config import Config
from backend.pipeline.preprocessing import decode_image_bytes


def is_allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in Config.ALLOWED_EXTENSIONS


def build_prefix_from_filename(filename: str | None, fallback: str) -> str:
    if not filename:
        return fallback
    stem = secure_filename(Path(filename).stem)
    return stem or fallback


def read_uploaded_image(file_storage: FileStorage) -> np.ndarray:
    if file_storage is None or not file_storage.filename:
        raise ValueError("Missing uploaded image.")

    if not is_allowed_file(file_storage.filename):
        allowed = ", ".join(sorted(Config.ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported image format. Allowed formats: {allowed}.")

    file_storage.stream.seek(0)
    file_bytes = file_storage.read()
    if not file_bytes:
        raise ValueError("Uploaded image is empty.")

    image = decode_image_bytes(file_bytes)
    file_storage.stream.seek(0)
    return image


def build_output_name(prefix: str, extension: str = ".png") -> str:
    safe_prefix = secure_filename(prefix) or "output"
    return f"{safe_prefix}_{uuid4().hex[:12]}{extension}"


def save_image(image: np.ndarray, prefix: str) -> str:
    output_name = build_output_name(prefix)
    output_path = Config.OUTPUT_DIR / output_name
    if not cv2.imwrite(str(output_path), image):
        raise ValueError("Failed to save output image.")
    return output_name


def to_asset_url(filename: str) -> str:
    return f"/assets/{filename}"


def serialize_bbox(bbox: tuple[int, int, int, int]) -> dict[str, int]:
    x, y, w, h = bbox
    return {
        "x": int(x),
        "y": int(y),
        "width": int(w),
        "height": int(h),
    }
