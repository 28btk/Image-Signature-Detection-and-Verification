from __future__ import annotations

import numpy as np


def summarize_pair_result(score: float, threshold: float) -> dict[str, object]:
    margin = score - threshold

    if abs(margin) >= 0.12:
        confidence_level = "high"
    elif abs(margin) >= 0.05:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    if score >= threshold:
        message = "The two signatures are considered to belong to the same writer."
    else:
        message = "The two signatures are considered visually inconsistent and likely forged."

    return {
        "margin": round(float(margin), 4),
        "confidence_level": confidence_level,
        "message": message,
    }


def evaluate_predictions(labels: list[int], scores: list[float], threshold: float) -> dict[str, object]:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length.")
    if not labels:
        raise ValueError("labels and scores cannot be empty.")

    y_true = np.asarray(labels, dtype=np.int32)
    y_pred = (np.asarray(scores, dtype=np.float32) >= threshold).astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(y_true)
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1_score), 4),
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
    }


def tune_threshold(labels: list[int], scores: list[float], search_start: float = 0.35, search_end: float = 0.95):
    thresholds = np.arange(search_start, search_end, 0.01)
    best_result = None

    for threshold in thresholds:
        metrics = evaluate_predictions(labels, scores, float(threshold))
        if best_result is None or metrics["f1_score"] > best_result["metrics"]["f1_score"]:
            best_result = {
                "best_threshold": round(float(threshold), 4),
                "metrics": metrics,
            }

    if best_result is None:
        raise ValueError("Unable to tune threshold with the provided data.")

    return best_result
