from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    FRONTEND_DIR = BASE_DIR / "frontend"
    STORAGE_DIR = BASE_DIR / "storage"
    UPLOAD_DIR = STORAGE_DIR / "uploads"
    OUTPUT_DIR = STORAGE_DIR / "outputs"

    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}

    NORMALIZED_WIDTH = 256
    NORMALIZED_HEIGHT = 128

    DETECTION_MIN_AREA_RATIO = 0.0008
    VERIFICATION_THRESHOLD = 0.70


def ensure_directories() -> None:
    for path in (Config.STORAGE_DIR, Config.UPLOAD_DIR, Config.OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)
