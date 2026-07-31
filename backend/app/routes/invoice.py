from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from flask import Blueprint, jsonify, request

from app.models.invoice import InvoiceStatus
from app.repositories.invoice_repository import InvoiceRepository
from app.services.extraction_service import AI_FIELDS, MONEY_FIELDS, ExtractionService
from extensions import db

invoice_bp = Blueprint("invoice", __name__)


def serialize_invoice(invoice, include_document=False):
    payload = {
        "id": invoice.id, "document_id": invoice.document_id, "supplier": invoice.supplier,
        "tax_identifier": invoice.tax_identifier, "invoice_number": invoice.invoice_number,
        "invoice_date": invoice.invoice_date.isoformat() if invoice.invoice_date else None,
        "total_ht": str(invoice.total_ht) if invoice.total_ht is not None else None,
        "vat_amount": str(invoice.vat_amount) if invoice.vat_amount is not None else None,
        "stamp_amount": str(invoice.stamp_amount) if invoice.stamp_amount is not None else None,
        "total_ttc": str(invoice.total_ttc) if invoice.total_ttc is not None else None,
        "currency": invoice.currency, "category": invoice.category, "status": invoice.status,
        "validated_at": invoice.validated_at.isoformat() if invoice.validated_at else None,
        "field_values": [{"field_name": x.field_name, "extracted_value": x.extracted_value,
                          "validated_value": x.validated_value, "confidence": x.confidence, "source": x.source}
                         for x in invoice.field_values],
    }
    if include_document:
        payload["document"] = {"id": invoice.document.id, "filename": invoice.document.original_name,
                               "status": invoice.document.status, "created_at": invoice.document.created_at.isoformat()}
    return payload


@invoice_bp.route("/api/invoices", methods=["GET"])
def list_invoices():
    invoices = InvoiceRepository().list_invoices()
    return jsonify({"invoices": [serialize_invoice(invoice, include_document=True) for invoice in invoices]})


@invoice_bp.route("/api/invoices/<int:invoice_id>", methods=["GET"])
def get_invoice(invoice_id):
    invoice = InvoiceRepository().get_invoice(invoice_id)
    if not invoice:
        return jsonify({"error": "Invoice not found"}), 404
    return jsonify({"invoice": serialize_invoice(invoice, include_document=True)})


@invoice_bp.route("/api/documents/<int:document_id>/process", methods=["POST"])
def process_document(document_id):
    document = InvoiceRepository().get_document(document_id)
    if not document:
        return jsonify({"error": "Document not found"}), 404
    try:
        invoice, result = ExtractionService().process_document(document)
        return jsonify({"invoice": serialize_invoice(invoice, include_document=True), "ai_result": result})
    except Exception as exc:
        return jsonify({"error": f"Processing failed: {exc}"}), 422


@invoice_bp.route("/api/invoices/<int:invoice_id>", methods=["PUT"])
def validate_invoice(invoice_id):
    invoice = InvoiceRepository().get_invoice(invoice_id)
    if not invoice:
        return jsonify({"error": "Invoice not found"}), 404
    data = request.get_json(silent=True) or {}
    for field in AI_FIELDS:
        if field not in data:
            continue
        value = data[field]
        try:
            if field in MONEY_FIELDS and value not in (None, ""):
                value = Decimal(str(value)).quantize(Decimal("0.001"))
            elif field in MONEY_FIELDS:
                value = None
            elif field == "invoice_date" and value:
                value = date.fromisoformat(value)
            elif field == "currency":
                value = str(value or "TND").upper()[:10]
        except (InvalidOperation, ValueError):
            return jsonify({"error": f"Invalid value for {field}"}), 400
        setattr(invoice, field, value)
        field_value = next((x for x in invoice.field_values if x.field_name == field), None)
        if field_value:
            field_value.validated_value = str(value) if value is not None else None
    invoice.status = InvoiceStatus.VALIDATED.value
    invoice.document.status = InvoiceStatus.VALIDATED.value
    invoice.validated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({"invoice": serialize_invoice(invoice, include_document=True)})
