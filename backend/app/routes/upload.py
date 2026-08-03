import os
from flask import Blueprint, jsonify, request, send_from_directory, current_app, abort
from werkzeug.utils import secure_filename

from app.services.upload_service import UploadService
from app.services.extraction_service import ExtractionService
from app.routes.invoice import serialize_invoice
from extensions import db

upload_bp = Blueprint("upload", __name__)


@upload_bp.route("/api/upload", methods=["POST"])
@upload_bp.route("/api/uploads", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "Aucun fichier sélectionné"}), 400
    
    try:
        document, invoice = UploadService().save_uploaded_file(file)
        db.session.commit()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  
        db.session.rollback()
        return jsonify({"error": f"Upload failed: {exc}"}), 500

    # Auto-process the document immediately to match Api/app.py behavior
    try:
        invoice, result = ExtractionService().process_document(document)
    except Exception as exc:
        return jsonify({
            "error": f"Erreur lors du traitement du document : {str(exc)}",
            "document_id": document.id,
            "invoice_id": invoice.id
        }), 500

    return jsonify({
        "message": "Fichier traité avec succès",
        "document_id": document.id,
        "invoice_id": invoice.id,
        "invoice": serialize_invoice(invoice, include_document=True),
        "status": document.status,
    }), 201


@upload_bp.route('/api/uploads/<path:filename>', methods=['GET'])
def serve_upload(filename):
    """Serve a previously uploaded document (image/PDF) for preview."""
    safe_name = secure_filename(filename)
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], safe_name)
    if not os.path.isfile(file_path):
        abort(404)
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], safe_name)
