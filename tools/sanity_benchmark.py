from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import Config
from backend.pipeline.evaluation import evaluate_predictions
from backend.pipeline.features import extract_signature_features
from backend.pipeline.normalization import normalize_signature
from backend.pipeline.verification_model import compare_signatures


def _make_signature(text: str, shift: tuple[int, int] = (0, 0), scale: float = 2.2, angle: float = 0.0) -> np.ndarray:
    image = np.full((150, 360, 3), 255, dtype=np.uint8)
    x_offset, y_offset = shift
    cv2.putText(
        image,
        text,
        (35 + x_offset, 94 + y_offset),
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    if abs(angle) < 1e-6:
        return image

    center = (image.shape[1] // 2, image.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderValue=(255, 255, 255))


def _score_pair(image_a: np.ndarray, image_b: np.ndarray) -> dict[str, float | str]:
    normalized_a = normalize_signature(image_a)
    normalized_b = normalize_signature(image_b)
    features_a = extract_signature_features(normalized_a["normalized_image"], normalized_a["ink_mask"])
    features_b = extract_signature_features(normalized_b["normalized_image"], normalized_b["ink_mask"])
    result = compare_signatures(
        features_a,
        features_b,
        normalized_a["ink_mask"],
        normalized_b["ink_mask"],
        threshold=Config.VERIFICATION_THRESHOLD,
    )
    return {
        "advanced_score": float(result["score"]),
        "baseline_score": float(result["baseline"]["score"]),
        "keypoint_method": str(result["metrics"]["keypoint_method"]),
        "keypoint_matches": int(result["metrics"]["keypoint_good_matches"]),
    }


def run_sanity_benchmark() -> dict[str, object]:
    positive_pairs = [
        ("Bui", (0, 0), (4, -2), 2.2, 2.2, 0.0, 0.0),
        ("Kiet", (0, 0), (-5, 3), 2.0, 2.05, 0.0, 1.5),
        ("Tuan", (0, 0), (3, 2), 2.0, 1.95, 0.0, -1.5),
        ("Vision", (0, 0), (0, 0), 2.0, 2.0, 0.0, -8.0),
        ("Sign", (0, 0), (0, 0), 2.0, 2.0, 0.0, 8.0),
        ("Verify", (0, 0), (-3, 3), 1.55, 1.58, 0.0, 0.0),
    ]
    negative_pairs = [
        ("Bui", "Nam", 2.2, 2.2),
        ("Kiet", "Long", 2.0, 2.0),
        ("Tuan", "Minh", 2.0, 2.0),
        ("Vision", "Forgery", 1.55, 1.45),
        ("Sign", "Check", 2.15, 1.9),
        ("Verify", "Writer", 1.55, 1.55),
    ]

    labels: list[int] = []
    advanced_scores: list[float] = []
    baseline_scores: list[float] = []
    rows: list[dict[str, object]] = []

    for index, (text, shift_a, shift_b, scale_a, scale_b, angle_a, angle_b) in enumerate(positive_pairs, start=1):
        image_a = _make_signature(text, shift_a, scale_a, angle_a)
        image_b = _make_signature(text, shift_b, scale_b, angle_b)
        result = _score_pair(image_a, image_b)
        labels.append(1)
        advanced_scores.append(float(result["advanced_score"]))
        baseline_scores.append(float(result["baseline_score"]))
        rows.append({"case": f"positive_{index}", "label": 1, **result})

    for index, (text_a, text_b, scale_a, scale_b) in enumerate(negative_pairs, start=1):
        image_a = _make_signature(text_a, (0, 0), scale_a, 0.0)
        image_b = _make_signature(text_b, (0, 0), scale_b, 0.0)
        result = _score_pair(image_a, image_b)
        labels.append(0)
        advanced_scores.append(float(result["advanced_score"]))
        baseline_scores.append(float(result["baseline_score"]))
        rows.append({"case": f"negative_{index}", "label": 0, **result})

    return {
        "description": "Deterministic internal sanity benchmark with synthetic signature pairs.",
        "threshold": Config.VERIFICATION_THRESHOLD,
        "pair_count": len(labels),
        "positive_pairs": len(positive_pairs),
        "negative_pairs": len(negative_pairs),
        "advanced_metrics": evaluate_predictions(labels, advanced_scores, Config.VERIFICATION_THRESHOLD),
        "baseline_metrics": evaluate_predictions(labels, baseline_scores, Config.VERIFICATION_THRESHOLD),
        "rows": rows,
    }


def main() -> None:
    result = run_sanity_benchmark()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
