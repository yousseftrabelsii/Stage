from flask import Blueprint, jsonify, request

from app.services.upload_service import UploadService
from extensions import db

upload_bp = Blueprint("upload", __name__)


@upload_bp.route("/api/uploads", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "A file is required"}), 400
    try:
        document, invoice = UploadService().save_uploaded_file(file)
        db.session.commit()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - storage failure
        db.session.rollback()
        return jsonify({"error": f"Upload failed: {exc}"}), 500
    return jsonify({
        "message": "Document uploaded",
        "document_id": document.id,
        "invoice_id": invoice.id,
        "status": document.status,
    }), 201
