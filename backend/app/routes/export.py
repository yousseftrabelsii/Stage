from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from app.models.invoice import Invoice, InvoiceStatus
from app.services.export_service import ExportService

export_bp = Blueprint("export", __name__)


@export_bp.route("/api/exports", methods=["GET"])
def export_data():
    export_format = request.args.get("format", "csv").lower()
    invoices = Invoice.query.filter(Invoice.status.in_([InvoiceStatus.VALIDATED.value, InvoiceStatus.EXPORTED.value])).all()
    if not invoices:
        return jsonify({"error": "No validated invoice is available for export"}), 400
    try:
        path = ExportService().export_invoices(invoices, export_format)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    mimetype = "text/csv" if export_format == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return send_file(Path(path), as_attachment=True, download_name=Path(path).name, mimetype=mimetype)
