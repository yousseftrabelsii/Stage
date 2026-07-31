from __future__ import annotations

from app.models.invoice import Document, Invoice
from extensions import db


class InvoiceRepository:
    def create_document(self, **values) -> Document:
        document = Document(**values)
        db.session.add(document)
        db.session.flush()
        return document

    def create_invoice(self, document: Document) -> Invoice:
        invoice = Invoice(document_id=document.id)
        db.session.add(invoice)
        db.session.flush()
        return invoice

    def get_document(self, document_id: int) -> Document | None:
        return db.session.get(Document, document_id)

    def get_invoice(self, invoice_id: int) -> Invoice | None:
        return db.session.get(Invoice, invoice_id)

    def list_invoices(self) -> list[Invoice]:
        return Invoice.query.order_by(Invoice.created_at.desc()).all()
