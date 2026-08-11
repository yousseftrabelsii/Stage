from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from app.models.invoice import Invoice, InvoiceStatus
from app.services.export_service import ExportService
from extensions import db

export_bp = Blueprint("export", __name__)


@export_bp.route("/api/exports", methods=["GET"])
def export_data_v2():
    """V2 parameterized export route (format=csv|xlsx)"""
    export_format = request.args.get("format", "csv").lower()
    return _do_export(export_format)


@export_bp.route("/api/export/excel", methods=["GET"])
def export_excel_legacy():
    """Legacy export route matching original API spec"""
    return _do_export("xlsx")


@export_bp.route("/api/export/csv", methods=["GET"])
def export_csv_legacy():
    """Legacy export route matching original API spec"""
    return _do_export("csv")


def _do_export(export_format: str):
    # Fetch all invoices that aren't failed/rejected.
    # The original API exported everything.
    invoices = Invoice.query.order_by(Invoice.created_at.desc()).all()
    if not invoices:
        return jsonify({"error": "Aucune facture disponible pour l'export"}), 400
        
    try:
        path = ExportService().export_invoices(invoices, export_format)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
        
    if export_format == "csv":
        mimetype = "text/csv" 
    else:
        mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
    return send_file(
        Path(path), 
        as_attachment=True, 
        download_name=Path(path).name, 
        mimetype=mimetype
    )
