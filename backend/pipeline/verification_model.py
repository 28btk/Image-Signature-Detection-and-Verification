from __future__ import annotations

import cv2
import numpy as np

from backend.config import Config
from backend.pipeline.features import SignatureFeatures, skeletonize_mask


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denominator = (np.linalg.norm(vector_a) * np.linalg.norm(vector_b)) + 1e-8
    return float(np.dot(vector_a, vector_b) / denominator)


def _same_size_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mask_a.shape[:2] == mask_b.shape[:2]:
        return (mask_a, mask_b)

    resized_b = cv2.resize(
        mask_b,
        (mask_a.shape[1], mask_a.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return (mask_a, resized_b)


def binary_overlap_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a, mask_b = _same_size_masks(mask_a, mask_b)
    foreground_a = mask_a > 0
    foreground_b = mask_b > 0
    intersection = np.logical_and(foreground_a, foreground_b).sum()
    union = np.logical_or(foreground_a, foreground_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _translate_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    transform = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        mask,
        transform,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )


def aligned_binary_overlap_score(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    search_radius: int = 6,
    step: int = 2,
) -> dict[str, object]:
    mask_a, mask_b = _same_size_masks(mask_a, mask_b)
    best_score = binary_overlap_score(mask_a, mask_b)
    best_offset = {"dx": 0, "dy": 0}

    for dy in range(-search_radius, search_radius + 1, step):
        for dx in range(-search_radius, search_radius + 1, step):
            if dx == 0 and dy == 0:
                continue
            shifted = _translate_mask(mask_b, dx, dy)
            score = binary_overlap_score(mask_a, shifted)
            if score > best_score:
                best_score = score
                best_offset = {"dx": dx, "dy": dy}

    return {
        "score": float(best_score),
        "offset": best_offset,
    }


def contour_shape_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a, mask_b = _same_size_masks(mask_a, mask_b)
    contours_a, _ = cv2.findContours(mask_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_b, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours_a or not contours_b:
        return 0.0

    contour_a = max(contours_a, key=cv2.contourArea)
    contour_b = max(contours_b, key=cv2.contourArea)
    distance = cv2.matchShapes(contour_a, contour_b, cv2.CONTOURS_MATCH_I1, 0.0)
    return float(1.0 / (1.0 + (distance * 5.0)))


def density_similarity(features_a: SignatureFeatures, features_b: SignatureFeatures) -> float:
    difference = abs(features_a.ink_ratio - features_b.ink_ratio)
    return float(max(0.0, 1.0 - (difference / 0.20)))


def chamfer_skeleton_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a, mask_b = _same_size_masks(mask_a, mask_b)
    skeleton_a = skeletonize_mask(mask_a)
    skeleton_b = skeletonize_mask(mask_b)

    if cv2.countNonZero(skeleton_a) == 0 or cv2.countNonZero(skeleton_b) == 0:
        return 0.0

    def mean_distance(source: np.ndarray, target: np.ndarray) -> float:
        source_foreground = source > 0
        target_distance_input = np.where(target > 0, 0, 255).astype(np.uint8)
        distance_map = cv2.distanceTransform(target_distance_input, cv2.DIST_L2, 3)
        return float(distance_map[source_foreground].mean())

    symmetric_distance = (mean_distance(skeleton_a, skeleton_b) + mean_distance(skeleton_b, skeleton_a)) / 2.0
    diagonal = float(np.hypot(mask_a.shape[0], mask_a.shape[1]))
    return float(np.exp(-symmetric_distance / ((0.04 * diagonal) + 1e-8)))


def _create_local_feature_extractor() -> tuple[str, object, int, float]:
    if hasattr(cv2, "SIFT_create"):
        return ("SIFT", cv2.SIFT_create(nfeatures=500, contrastThreshold=0.015, edgeThreshold=8), cv2.NORM_L2, 0.75)
    return ("ORB", cv2.ORB_create(nfeatures=500, scaleFactor=1.2, edgeThreshold=5, patchSize=15, fastThreshold=7), cv2.NORM_HAMMING, 0.78)


def keypoint_geometric_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, object]:
    mask_a, mask_b = _same_size_masks(mask_a, mask_b)
    image_a = cv2.GaussianBlur(mask_a, (3, 3), 0)
    image_b = cv2.GaussianBlur(mask_b, (3, 3), 0)

    method, extractor, norm_type, ratio_threshold = _create_local_feature_extractor()
    keypoints_a, descriptors_a = extractor.detectAndCompute(image_a, None)
    keypoints_b, descriptors_b = extractor.detectAndCompute(image_b, None)
    keypoints_a = keypoints_a or []
    keypoints_b = keypoints_b or []

    if descriptors_a is None or descriptors_b is None or len(keypoints_a) < 2 or len(keypoints_b) < 2:
        return {
            "method": method,
            "score": 0.0,
            "reliability": 0.0,
            "keypoints_a": len(keypoints_a),
            "keypoints_b": len(keypoints_b),
            "good_matches": 0,
            "inlier_ratio": 0.0,
        }

    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    raw_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good_matches = []
    for match_pair in raw_matches:
        if len(match_pair) < 2:
            continue
        best, second_best = match_pair
        if best.distance < ratio_threshold * second_best.distance:
            good_matches.append(best)

    inlier_ratio = 0.0
    if len(good_matches) >= 4:
        source_points = np.float32([keypoints_a[item.queryIdx].pt for item in good_matches]).reshape(-1, 1, 2)
        target_points = np.float32([keypoints_b[item.trainIdx].pt for item in good_matches]).reshape(-1, 1, 2)
        try:
            _, inliers = cv2.findHomography(source_points, target_points, cv2.RANSAC, 5.0)
        except cv2.error:
            inliers = None
        if inliers is not None:
            inlier_ratio = float(inliers.ravel().sum() / max(len(good_matches), 1))

    match_ratio = len(good_matches) / max(min(len(keypoints_a), len(keypoints_b)), 1)
    boosted_match_ratio = min(match_ratio * 2.0, 1.0)
    score = min((0.55 * boosted_match_ratio) + (0.45 * inlier_ratio), 1.0)
    reliability = min(len(good_matches) / 10.0, 1.0)
    if len(good_matches) < 4:
        reliability *= 0.5

    return {
        "method": method,
        "score": float(score),
        "reliability": float(reliability),
        "keypoints_a": len(keypoints_a),
        "keypoints_b": len(keypoints_b),
        "good_matches": len(good_matches),
        "inlier_ratio": float(inlier_ratio),
    }


def _prediction_from_score(score: float, threshold: float) -> tuple[str, float]:
    prediction = "genuine" if score >= threshold else "forged"
    confidence = min(abs(score - threshold) / 0.25, 1.0)
    return (prediction, confidence)


def _weighted_average(components: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in components if weight > 0)
    if total_weight <= 0:
        return 0.0
    return float(sum(score * weight for score, weight in components if weight > 0) / total_weight)


def compare_baseline_signatures(
    features_a: SignatureFeatures,
    features_b: SignatureFeatures,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    threshold: float | None = None,
) -> dict[str, object]:
    threshold = Config.VERIFICATION_THRESHOLD if threshold is None else threshold

    feature_score = cosine_similarity(features_a.classic_vector, features_b.classic_vector)
    overlap_score = binary_overlap_score(mask_a, mask_b)
    contour_score = contour_shape_similarity(mask_a, mask_b)
    density_score = density_similarity(features_a, features_b)

    final_score = (
        (0.45 * feature_score)
        + (0.25 * overlap_score)
        + (0.15 * contour_score)
        + (0.15 * density_score)
    )

    prediction, confidence = _prediction_from_score(final_score, threshold)

    return {
        "prediction": prediction,
        "score": round(float(final_score), 4),
        "threshold": round(float(threshold), 4),
        "confidence": round(float(confidence), 4),
        "metrics": {
            "feature_similarity": round(float(feature_score), 4),
            "overlap_similarity": round(float(overlap_score), 4),
            "contour_similarity": round(float(contour_score), 4),
            "density_similarity": round(float(density_score), 4),
        },
    }


def compare_signatures(
    features_a: SignatureFeatures,
    features_b: SignatureFeatures,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    threshold: float | None = None,
) -> dict[str, object]:
    threshold = Config.VERIFICATION_THRESHOLD if threshold is None else threshold

    baseline = compare_baseline_signatures(features_a, features_b, mask_a, mask_b, threshold)

    feature_score = cosine_similarity(features_a.vector, features_b.vector)
    exact_overlap_score = binary_overlap_score(mask_a, mask_b)
    aligned_overlap = aligned_binary_overlap_score(mask_a, mask_b)
    contour_score = contour_shape_similarity(mask_a, mask_b)
    density_score = density_similarity(features_a, features_b)
    chamfer_score = chamfer_skeleton_similarity(mask_a, mask_b)
    keypoint_score = keypoint_geometric_similarity(mask_a, mask_b)

    keypoint_weight = 0.12 * float(keypoint_score["reliability"])
    final_score = _weighted_average(
        [
            (feature_score, 0.32),
            (float(aligned_overlap["score"]), 0.18),
            (contour_score, 0.10),
            (density_score, 0.08),
            (chamfer_score, 0.20),
            (float(keypoint_score["score"]), keypoint_weight),
        ]
    )

    prediction, confidence = _prediction_from_score(final_score, threshold)
    score_delta = final_score - float(baseline["score"])

    return {
        "prediction": prediction,
        "score": round(float(final_score), 4),
        "threshold": round(float(threshold), 4),
        "confidence": round(float(confidence), 4),
        "metrics": {
            "feature_similarity": round(float(feature_score), 4),
            "overlap_similarity": round(float(aligned_overlap["score"]), 4),
            "exact_overlap_similarity": round(float(exact_overlap_score), 4),
            "contour_similarity": round(float(contour_score), 4),
            "density_similarity": round(float(density_score), 4),
            "chamfer_similarity": round(float(chamfer_score), 4),
            "keypoint_similarity": round(float(keypoint_score["score"]), 4),
            "keypoint_reliability": round(float(keypoint_score["reliability"]), 4),
            "keypoint_method": str(keypoint_score["method"]),
            "keypoint_good_matches": int(keypoint_score["good_matches"]),
            "keypoint_inlier_ratio": round(float(keypoint_score["inlier_ratio"]), 4),
            "alignment_offset": aligned_overlap["offset"],
        },
        "baseline": {
            "prediction": baseline["prediction"],
            "score": baseline["score"],
            "threshold": baseline["threshold"],
            "metrics": baseline["metrics"],
        },
        "improvement": {
            "model": "Advanced CV ensemble: HOG + skeleton Chamfer + aligned IoU + SIFT/ORB-RANSAC",
            "score_delta": round(float(score_delta), 4),
            "active_keypoint_weight": round(float(keypoint_weight), 4),
        },
    }
