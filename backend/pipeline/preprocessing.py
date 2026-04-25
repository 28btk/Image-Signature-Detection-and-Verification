from __future__ import annotations

import cv2
import numpy as np


def decode_image_bytes(file_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a valid image.")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_contrast(gray: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray)


def remove_document_noise(gray: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(gray, (5, 5), 0)


def adaptive_binary_mask(gray: np.ndarray) -> np.ndarray:
    enhanced = normalize_contrast(gray)
    return cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        11,
    )


def suppress_document_lines(binary_mask: np.ndarray) -> np.ndarray:
    image_height, image_width = binary_mask.shape[:2]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, image_width // 8), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, image_height // 8)))

    horizontal_lines = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical_lines = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    detected_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
    return cv2.subtract(binary_mask, detected_lines)


def clean_detection_mask(binary_mask: np.ndarray) -> np.ndarray:
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))

    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    connected = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, connect_kernel, iterations=2)
    return connected


def preprocess_document_image(image: np.ndarray) -> dict[str, np.ndarray]:
    gray = to_grayscale(image)
    denoised = remove_document_noise(gray)
    binary = adaptive_binary_mask(denoised)
    line_suppressed = suppress_document_lines(binary)
    cleaned = clean_detection_mask(line_suppressed)

    return {
        "gray": gray,
        "binary": binary,
        "line_suppressed": line_suppressed,
        "mask": cleaned,
    }


def preprocess_signature_image(image: np.ndarray) -> dict[str, np.ndarray]:
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    return {
        "gray": gray,
        "binary": cleaned,
    }
