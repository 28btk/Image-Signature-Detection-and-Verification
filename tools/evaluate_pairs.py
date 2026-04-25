from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import Config
from backend.pipeline.evaluation import evaluate_predictions, tune_threshold
from backend.pipeline.features import extract_signature_features
from backend.pipeline.normalization import normalize_signature
from backend.pipeline.verification_model import compare_signatures


def _read_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return base_dir / path


def _compare_pair(signature_a: Path, signature_b: Path, threshold: float) -> dict[str, object]:
    image_a = _read_image(signature_a)
    image_b = _read_image(signature_b)

    normalized_a = normalize_signature(image_a)
    normalized_b = normalize_signature(image_b)
    features_a = extract_signature_features(normalized_a["normalized_image"], normalized_a["ink_mask"])
    features_b = extract_signature_features(normalized_b["normalized_image"], normalized_b["ink_mask"])

    result = compare_signatures(
        features_a,
        features_b,
        normalized_a["ink_mask"],
        normalized_b["ink_mask"],
        threshold=threshold,
    )
    return {
        "advanced_score": result["score"],
        "baseline_score": result["baseline"]["score"],
        "advanced_prediction": result["prediction"],
        "baseline_prediction": result["baseline"]["prediction"],
        "score_delta": result["improvement"]["score_delta"],
    }


def evaluate_pair_csv(csv_path: Path, image_root: Path, threshold: float) -> dict[str, object]:
    labels: list[int] = []
    advanced_scores: list[float] = []
    baseline_scores: list[float] = []
    rows: list[dict[str, object]] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"signature_a", "signature_b", "label"}
        if not required_columns.issubset(reader.fieldnames or []):
            columns = ", ".join(sorted(required_columns))
            raise ValueError(f"CSV must contain columns: {columns}")

        for row_index, row in enumerate(reader, start=1):
            label = int(row["label"])
            signature_a = _resolve_path(image_root, row["signature_a"])
            signature_b = _resolve_path(image_root, row["signature_b"])
            comparison = _compare_pair(signature_a, signature_b, threshold)

            labels.append(label)
            advanced_scores.append(float(comparison["advanced_score"]))
            baseline_scores.append(float(comparison["baseline_score"]))
            rows.append(
                {
                    "row": row_index,
                    "signature_a": str(signature_a),
                    "signature_b": str(signature_b),
                    "label": label,
                    **comparison,
                }
            )

    return {
        "threshold": threshold,
        "pair_count": len(labels),
        "advanced_metrics": evaluate_predictions(labels, advanced_scores, threshold),
        "baseline_metrics": evaluate_predictions(labels, baseline_scores, threshold),
        "advanced_threshold_tuning": tune_threshold(labels, advanced_scores),
        "baseline_threshold_tuning": tune_threshold(labels, baseline_scores),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and improved signature verification on a pair CSV.")
    parser.add_argument("--pairs", required=True, type=Path, help="CSV with columns: signature_a, signature_b, label.")
    parser.add_argument("--image-root", type=Path, default=Path("."), help="Base directory for relative image paths.")
    parser.add_argument("--threshold", type=float, default=Config.VERIFICATION_THRESHOLD)
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_pair_csv(args.pairs, args.image_root, args.threshold)
    payload = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
