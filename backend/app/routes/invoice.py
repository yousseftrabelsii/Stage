from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from flask import Blueprint, jsonify, request

from app.models.invoice import InvoiceStatus
from app.repositories.invoice_repository import InvoiceRepository
from app.services.extraction_service import AI_FIELDS, MONEY_FIELDS, ExtractionService
from extensions import db

invoice_bp = Blueprint("invoice", __name__)


def serialize_invoice(invoice, include_document=False):
    # Use the `to_dict` method we added to the Invoice model
    payload = invoice.to_dict()
    
    # Add field validation/confidence details
    payload["field_values"] = [
        {
            "field_name": x.field_name, 
            "extracted_value": x.extracted_value,
            "validated_value": x.validated_value, 
            "confidence": x.confidence, 
            "source": x.source
        } for x in invoice.field_values
    ]
    
    if include_document and invoice.document:
        payload["document"] = {
            "id": invoice.document.id, 
            "filename": invoice.document.original_name,
            "status": invoice.document.status, 
            "created_at": invoice.document.created_at.isoformat()
        }
    return payload


@invoice_bp.route("/api/invoices", methods=["GET"])
def list_invoices():
    invoices = InvoiceRepository().list_invoices()
    return jsonify([serialize_invoice(invoice, include_document=False) for invoice in invoices])


@invoice_bp.route("/api/invoices/<int:invoice_id>", methods=["GET"])
def get_invoice(invoice_id):
    invoice = InvoiceRepository().get_invoice(invoice_id)
    if not invoice:
        return jsonify({"error": "Facture introuvable"}), 404
    return jsonify(serialize_invoice(invoice, include_document=True))


@invoice_bp.route("/api/documents/<int:document_id>/process", methods=["POST"])
def process_document(document_id):
    document = InvoiceRepository().get_document(document_id)
    if not document:
        return jsonify({"error": "Document introuvable"}), 404
    try:
        invoice, result = ExtractionService().process_document(document)
        return jsonify({
            "message": "Extraction terminée",
            "invoice": serialize_invoice(invoice, include_document=True), 
            "ai_result": result
        })
    except Exception as exc:
        return jsonify({"error": f"Erreur lors du traitement: {exc}"}), 500


@invoice_bp.route("/api/invoices/<int:invoice_id>", methods=["PUT"])
def validate_invoice(invoice_id):
    invoice = InvoiceRepository().get_invoice(invoice_id)
    if not invoice:
        return jsonify({"error": "Facture introuvable"}), 404
    
    data = request.get_json(silent=True) or {}
    for field in AI_FIELDS:
        if field not in data:
            continue
        value = data[field]
        
        # Handle field-specific conversions
        try:
            # Special case for vat -> vat_amount aliasing in frontend
            if field == "vat_amount" and "vat" in data:
                value = data["vat"]
                
            if field in MONEY_FIELDS and value not in (None, ""):
                value = Decimal(str(value)).quantize(Decimal("0.001"))
            elif field in MONEY_FIELDS:
                value = None
            elif field == "invoice_date" and value:
                value = date.fromisoformat(value)
            elif field == "currency":
                value = str(value or "TND").upper()[:10]
        except (InvalidOperation, ValueError):
            return jsonify({"error": f"Valeur invalide pour {field}"}), 400
            
        setattr(invoice, field, value)
        
        field_value = next((x for x in invoice.field_values if x.field_name == field), None)
        if field_value:
            field_value.validated_value = str(value) if value is not None else None
            
    invoice.status = data.get("status", InvoiceStatus.VALIDATED.value)
    if invoice.document:
        invoice.document.status = invoice.status
    invoice.validated_at = datetime.utcnow()
    
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Erreur lors de la sauvegarde: {str(e)}"}), 500
        
    return jsonify(serialize_invoice(invoice, include_document=False))


@invoice_bp.route('/api/invoices/<int:invoice_id>', methods=['DELETE'])
def delete_invoice(invoice_id):
    repo = InvoiceRepository()
    invoice = repo.get_invoice(invoice_id)
    if not invoice:
        return jsonify({"error": "Facture introuvable"}), 404
    try:
        repo.delete_invoice(invoice)
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la suppression : {str(e)}"}), 500
    return jsonify({"message": "Facture supprimée avec succès"}), 200
