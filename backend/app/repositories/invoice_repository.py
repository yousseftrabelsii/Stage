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

    def delete_invoice(self, invoice: Invoice) -> None:
        # Document is cascaded from Invoice? 
        # Actually in models: Document has invoice = db.relationship(..., cascade="all, delete-orphan")
        # So deleting Document deletes Invoice. Wait, no. Document is the parent.
        # Invoice has document = db.relationship... wait, Invoice has document_id = db.ForeignKey('documents.id').
        # Let's delete the document, which will cascade to invoice and extraction runs.
        if invoice.document:
            db.session.delete(invoice.document)
        else:
            db.session.delete(invoice)
        db.session.commit()
