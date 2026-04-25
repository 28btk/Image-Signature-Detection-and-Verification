from __future__ import annotations

import cv2
import numpy as np
from werkzeug.datastructures import FileStorage

from backend.pipeline.evaluation import summarize_pair_result
from backend.pipeline.features import build_feature_summary, extract_signature_features
from backend.pipeline.normalization import normalize_signature
from backend.pipeline.verification_model import compare_signatures
from backend.utils.file_utils import (
    build_prefix_from_filename,
    read_uploaded_image,
    save_image,
    to_asset_url,
)


def _build_comparison_preview(signature_a: np.ndarray, signature_b: np.ndarray) -> np.ndarray:
    left = cv2.cvtColor(signature_a, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(signature_b, cv2.COLOR_GRAY2BGR)

    canvas_height = max(left.shape[0], right.shape[0]) + 32
    canvas_width = left.shape[1] + right.shape[1] + 36
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

    canvas[16 : 16 + left.shape[0], 12 : 12 + left.shape[1]] = left
    canvas[16 : 16 + right.shape[0], 24 + left.shape[1] : 24 + left.shape[1] + right.shape[1]] = right
    cv2.line(
        canvas,
        (left.shape[1] + 18, 10),
        (left.shape[1] + 18, canvas_height - 10),
        (180, 180, 180),
        2,
    )
    cv2.putText(canvas, "A", (18, canvas_height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.putText(
        canvas,
        "B",
        (left.shape[1] + 30, canvas_height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        2,
    )
    return canvas


def run_signature_verification(
    signature_a_file: FileStorage,
    signature_b_file: FileStorage,
    threshold: float | None = None,
) -> dict[str, object]:
    signature_a = read_uploaded_image(signature_a_file)
    signature_b = read_uploaded_image(signature_b_file)

    normalized_a = normalize_signature(signature_a)
    normalized_b = normalize_signature(signature_b)

    features_a = extract_signature_features(
        normalized_a["normalized_image"],
        normalized_a["ink_mask"],
    )
    features_b = extract_signature_features(
        normalized_b["normalized_image"],
        normalized_b["ink_mask"],
    )

    verification = compare_signatures(
        features_a,
        features_b,
        normalized_a["ink_mask"],
        normalized_b["ink_mask"],
        threshold=threshold,
    )
    evaluation = summarize_pair_result(verification["score"], verification["threshold"])
    comparison_preview = _build_comparison_preview(
        normalized_a["normalized_image"],
        normalized_b["normalized_image"],
    )

    prefix_a = build_prefix_from_filename(signature_a_file.filename, "signature_a")
    prefix_b = build_prefix_from_filename(signature_b_file.filename, "signature_b")

    normalized_a_name = save_image(normalized_a["normalized_image"], f"{prefix_a}_normalized")
    normalized_b_name = save_image(normalized_b["normalized_image"], f"{prefix_b}_normalized")
    comparison_name = save_image(comparison_preview, f"{prefix_a}_{prefix_b}_compare")

    return {
        "mode": "verification",
        "pipeline": [
            "Preprocessing",
            "Signature Normalization",
            "Enhanced Feature Extraction",
            "Advanced Ensemble Verification Model",
            "Evaluation",
            "Baseline Comparison",
        ],
        "verification": {
            "prediction": verification["prediction"],
            "score": verification["score"],
            "threshold": verification["threshold"],
            "confidence": verification["confidence"],
            "metrics": verification["metrics"],
            "baseline": verification["baseline"],
            "improvement": verification["improvement"],
            "evaluation": evaluation,
        },
        "features": {
            "signature_a": build_feature_summary(features_a),
            "signature_b": build_feature_summary(features_b),
        },
        "outputs": {
            "normalized_signature_a": to_asset_url(normalized_a_name),
            "normalized_signature_b": to_asset_url(normalized_b_name),
            "comparison_preview": to_asset_url(comparison_name),
        },
    }
