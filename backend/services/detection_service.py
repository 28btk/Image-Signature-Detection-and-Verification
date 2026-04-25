from __future__ import annotations

from werkzeug.datastructures import FileStorage

from backend.pipeline.detection import detect_signature_region
from backend.pipeline.normalization import normalize_signature
from backend.utils.file_utils import (
    build_prefix_from_filename,
    read_uploaded_image,
    save_image,
    serialize_bbox,
    to_asset_url,
)


def run_signature_detection(file_storage: FileStorage) -> dict[str, object]:
    image = read_uploaded_image(file_storage)
    detection = detect_signature_region(image)
    normalized = normalize_signature(detection["cropped_signature"])

    prefix = build_prefix_from_filename(file_storage.filename, "document")
    mask_name = save_image(detection["mask"], f"{prefix}_mask")
    annotated_name = save_image(detection["annotated_image"], f"{prefix}_detected")
    cropped_name = save_image(detection["cropped_signature"], f"{prefix}_cropped")
    normalized_name = save_image(normalized["normalized_image"], f"{prefix}_normalized")

    return {
        "mode": "detection",
        "pipeline": [
            "Preprocessing",
            "Signature Detection / Cropping",
            "Signature Normalization",
        ],
        "detection": {
            "bbox": serialize_bbox(detection["bbox"]),
            "candidate_score": detection["candidate_score"],
            "candidate_count": detection["candidate_count"],
            "fill_ratio": detection["fill_ratio"],
            "contour_area": detection["contour_area"],
            "normalized_source_bbox": serialize_bbox(normalized["source_bbox"]),
        },
        "outputs": {
            "mask_image": to_asset_url(mask_name),
            "annotated_image": to_asset_url(annotated_name),
            "cropped_signature": to_asset_url(cropped_name),
            "normalized_signature": to_asset_url(normalized_name),
        },
    }
