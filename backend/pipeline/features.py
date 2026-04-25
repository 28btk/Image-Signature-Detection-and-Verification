from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SignatureFeatures:
    vector: np.ndarray
    classic_vector: np.ndarray
    ink_ratio: float
    aspect_ratio: float
    centroid_x: float
    centroid_y: float
    hu_moments: list[float]
    skeleton_density: float
    stroke_endpoints: int
    stroke_branchpoints: int


def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Could not find a valid contour in the normalized signature.")
    return max(contours, key=cv2.contourArea)


def _resample_profile(profile: np.ndarray, output_length: int) -> np.ndarray:
    values = profile.astype(np.float32).reshape(1, -1)
    resized = cv2.resize(values, (output_length, 1), interpolation=cv2.INTER_AREA)
    return resized.flatten()


def _signed_log_hu(hu_moments: np.ndarray) -> list[float]:
    stabilized = []
    for value in hu_moments:
        signed = np.sign(value) * np.log10(abs(value) + 1e-12)
        stabilized.append(float(signed))
    return stabilized


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # The mask is already normalized, so a bounded morphology loop is fast and deterministic.
    max_iterations = max(binary.shape[:2]) * 2
    for _ in range(max_iterations):
        if cv2.countNonZero(binary) == 0:
            break

        eroded = cv2.erode(binary, element)
        opened = cv2.dilate(eroded, element)
        residue = cv2.subtract(binary, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        binary = eroded

    return skeleton


def _count_skeleton_nodes(skeleton: np.ndarray) -> tuple[int, int]:
    binary = (skeleton > 0).astype(np.uint8)
    if cv2.countNonZero(binary) == 0:
        return (0, 0)

    padded = np.pad(binary, 1, mode="constant")
    y_values, x_values = np.where(binary > 0)
    endpoints = 0
    branchpoints = 0

    for y, x in zip(y_values, x_values):
        neighbors = int(padded[y : y + 3, x : x + 3].sum()) - 1
        if neighbors == 1:
            endpoints += 1
        elif neighbors >= 3:
            branchpoints += 1

    return (endpoints, branchpoints)


def _extract_hog_descriptor(mask: np.ndarray, cell_size: int = 16, bins: int = 9) -> np.ndarray:
    normalized_mask = mask.astype(np.float32) / 255.0
    gradient_x = cv2.Sobel(normalized_mask, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(normalized_mask, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)
    angle = np.mod(angle, 180.0)

    image_height, image_width = mask.shape[:2]
    cell_rows = image_height // cell_size
    cell_cols = image_width // cell_size
    descriptor = np.zeros((cell_rows, cell_cols, bins), dtype=np.float32)
    bin_width = 180.0 / bins

    for row in range(cell_rows):
        for col in range(cell_cols):
            y1 = row * cell_size
            x1 = col * cell_size
            cell_magnitude = magnitude[y1 : y1 + cell_size, x1 : x1 + cell_size]
            cell_angle = angle[y1 : y1 + cell_size, x1 : x1 + cell_size]
            bin_indices = np.floor(cell_angle / bin_width).astype(np.int32)
            bin_indices = np.clip(bin_indices, 0, bins - 1)
            np.add.at(descriptor[row, col], bin_indices.ravel(), cell_magnitude.ravel())

    descriptor = descriptor.flatten()
    return descriptor / (np.linalg.norm(descriptor) + 1e-8)


def extract_signature_features(normalized_image: np.ndarray, ink_mask: np.ndarray | None = None) -> SignatureFeatures:
    if ink_mask is None:
        ink_mask = 255 - normalized_image

    ink_pixels = cv2.countNonZero(ink_mask)
    if ink_pixels < 25:
        raise ValueError("Not enough signature pixels to extract features.")

    image_height, image_width = ink_mask.shape[:2]
    contour = _largest_contour(ink_mask)
    moments = cv2.moments(ink_mask)
    hu_moments = _signed_log_hu(cv2.HuMoments(moments).flatten())

    x, y, w, h = cv2.boundingRect(contour)
    centroid_x = (moments["m10"] / moments["m00"] / image_width) if moments["m00"] else 0.5
    centroid_y = (moments["m01"] / moments["m00"] / image_height) if moments["m00"] else 0.5
    ink_ratio = ink_pixels / float(image_width * image_height)
    aspect_ratio = w / max(h, 1)
    bbox_ratio = (w * h) / float(image_width * image_height)

    horizontal_profile = _resample_profile(
        ink_mask.sum(axis=1) / (255.0 * image_width),
        24,
    )
    vertical_profile = _resample_profile(
        ink_mask.sum(axis=0) / (255.0 * image_height),
        48,
    )
    low_resolution_shape = cv2.resize(ink_mask, (64, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low_resolution_shape = low_resolution_shape.flatten() / 255.0
    skeleton_mask = skeletonize_mask(ink_mask)
    skeleton_pixels = cv2.countNonZero(skeleton_mask)
    skeleton_density = skeleton_pixels / float(image_width * image_height)
    stroke_endpoints, stroke_branchpoints = _count_skeleton_nodes(skeleton_mask)
    low_resolution_skeleton = cv2.resize(skeleton_mask, (64, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low_resolution_skeleton = low_resolution_skeleton.flatten() / 255.0
    hog_descriptor = _extract_hog_descriptor(ink_mask)

    scalar_features = np.array(
        [
            ink_ratio,
            bbox_ratio,
            aspect_ratio / 10.0,
            centroid_x,
            centroid_y,
        ],
        dtype=np.float32,
    )
    stroke_features = np.array(
        [
            skeleton_density,
            stroke_endpoints / 200.0,
            stroke_branchpoints / 800.0,
        ],
        dtype=np.float32,
    )

    classic_vector = np.concatenate(
        [
            low_resolution_shape,
            horizontal_profile.astype(np.float32),
            vertical_profile.astype(np.float32),
            np.array(hu_moments, dtype=np.float32),
            scalar_features,
        ]
    ).astype(np.float32)
    classic_vector = classic_vector / (np.linalg.norm(classic_vector) + 1e-8)

    vector = np.concatenate(
        [
            low_resolution_shape,
            low_resolution_skeleton,
            hog_descriptor,
            horizontal_profile.astype(np.float32),
            vertical_profile.astype(np.float32),
            np.array(hu_moments, dtype=np.float32),
            scalar_features,
            stroke_features,
        ]
    ).astype(np.float32)
    vector = vector / (np.linalg.norm(vector) + 1e-8)

    return SignatureFeatures(
        vector=vector,
        classic_vector=classic_vector,
        ink_ratio=float(ink_ratio),
        aspect_ratio=float(aspect_ratio),
        centroid_x=float(centroid_x),
        centroid_y=float(centroid_y),
        hu_moments=hu_moments,
        skeleton_density=float(skeleton_density),
        stroke_endpoints=int(stroke_endpoints),
        stroke_branchpoints=int(stroke_branchpoints),
    )


def build_feature_summary(features: SignatureFeatures) -> dict[str, object]:
    return {
        "ink_ratio": round(features.ink_ratio, 4),
        "aspect_ratio": round(features.aspect_ratio, 4),
        "centroid_x": round(features.centroid_x, 4),
        "centroid_y": round(features.centroid_y, 4),
        "skeleton_density": round(features.skeleton_density, 4),
        "stroke_endpoints": features.stroke_endpoints,
        "stroke_branchpoints": features.stroke_branchpoints,
        "hu_moments": [round(value, 4) for value in features.hu_moments],
    }
