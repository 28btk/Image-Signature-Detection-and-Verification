from http import HTTPStatus

from flask import Blueprint, jsonify, request

from backend.services.detection_service import run_signature_detection
from backend.services.verification_service import run_signature_verification

api_bp = Blueprint("api", __name__)


def error_response(message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST):
    return jsonify({"success": False, "error": message}), status


@api_bp.get("/health")
def health_check():
    return jsonify(
        {
            "success": True,
            "message": "Signature service is ready.",
        }
    )


@api_bp.post("/detect")
def detect_signature():
    document = request.files.get("document")
    if document is None or document.filename == "":
        return error_response("Please upload a document image in field 'document'.")

    try:
        result = run_signature_detection(document)
    except ValueError as exc:
        return error_response(str(exc))
    except Exception as exc:
        return error_response(f"Detection failed: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    return jsonify({"success": True, **result})


@api_bp.post("/verify")
def verify_signature():
    signature_a = request.files.get("signature_a")
    signature_b = request.files.get("signature_b")

    if signature_a is None or signature_a.filename == "":
        return error_response("Please upload the first signature in field 'signature_a'.")
    if signature_b is None or signature_b.filename == "":
        return error_response("Please upload the second signature in field 'signature_b'.")

    threshold_raw = request.form.get("threshold", "").strip()
    threshold = None
    if threshold_raw:
        try:
            threshold = float(threshold_raw)
        except ValueError:
            return error_response("Threshold must be a float number, for example 0.70.")

    try:
        result = run_signature_verification(signature_a, signature_b, threshold=threshold)
    except ValueError as exc:
        return error_response(str(exc))
    except Exception as exc:
        return error_response(f"Verification failed: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    return jsonify({"success": True, **result})
