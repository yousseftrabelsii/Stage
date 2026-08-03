from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum

from sqlalchemy.dialects.postgresql import JSONB

from extensions import db


class InvoiceStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    EXTRACTED = "extracted"
    NEEDS_REVIEW = "needs_review"
    VALIDATED = "validated"
    EXPORTED = "exported"
    FAILED = "failed"
    REJECTED = "rejected"


class Document(db.Model):
    __tablename__ = "documents"

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)       # generated storage filename (uuid)
    original_name = db.Column(db.String(255), nullable=False)  # original user filename
    storage_path = db.Column(db.String(500), nullable=False)
    content_type = db.Column(db.String(100), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    file_hash = db.Column(db.String(64), nullable=False, index=True)
    document_type = db.Column(db.String(50), nullable=False, default="invoice")
    status = db.Column(db.String(50), nullable=False, default=InvoiceStatus.UPLOADED.value)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    invoice = db.relationship("Invoice", back_populates="document", uselist=False, cascade="all, delete-orphan")
    extraction_runs = db.relationship("ExtractionRun", back_populates="document", cascade="all, delete-orphan")


class ExtractionRun(db.Model):
    __tablename__ = "extraction_runs"

    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey("documents.id"), nullable=False, index=True)
    ocr_text = db.Column(db.Text, nullable=True)
    ai_model = db.Column(db.String(100), nullable=True)
    # JSONB on PostgreSQL; JSON fallback for SQLite-based tests.
    ai_raw_json = db.Column(db.JSON().with_variant(JSONB, "postgresql"), nullable=True)
    success = db.Column(db.Boolean, nullable=False, default=False)
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    document = db.relationship("Document", back_populates="extraction_runs")


class Invoice(db.Model):
    __tablename__ = "invoices"

    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey("documents.id"), nullable=False, unique=True)

    # ── Supplier & identifiers ─────────────────────────────────────────────────
    supplier = db.Column(db.String(255), nullable=True)
    tax_identifier = db.Column(db.String(100), nullable=True)
    invoice_number = db.Column(db.String(100), nullable=True)
    invoice_date = db.Column(db.Date, nullable=True)

    # ── Amounts ───────────────────────────────────────────────────────────────
    total_ht = db.Column(db.Numeric(12, 3), nullable=True)
    vat_amount = db.Column(db.Numeric(12, 3), nullable=True)   # TVA
    stamp_amount = db.Column(db.Numeric(12, 3), nullable=True) # Timbre fiscal (Tunisia)
    total_ttc = db.Column(db.Numeric(12, 3), nullable=True)
    currency = db.Column(db.String(10), nullable=False, default="TND")

    # ── Classification ────────────────────────────────────────────────────────
    # document_type: "Facture" | "Reçu" | "Avoir" | "Note" | "Autre"
    document_type = db.Column(db.String(100), nullable=True, default="Facture")
    # category: free-form business category / sector tag
    category = db.Column(db.String(100), nullable=True)

    # ── Workflow ──────────────────────────────────────────────────────────────
    status = db.Column(db.String(50), nullable=False, default=InvoiceStatus.UPLOADED.value)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    validated_at = db.Column(db.DateTime, nullable=True)

    document = db.relationship("Document", back_populates="invoice")
    field_values = db.relationship("InvoiceFieldValue", back_populates="invoice", cascade="all, delete-orphan")
    exports = db.relationship("ExportHistory", back_populates="invoice")

    def to_dict(self) -> dict:
        """Flat serialisation compatible with the frontend and the Api/ shape."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "filename": self.document.filename if self.document else None,
            "original_name": self.document.original_name if self.document else None,
            "supplier": self.supplier,
            "tax_identifier": self.tax_identifier,
            "invoice_number": self.invoice_number,
            "invoice_date": self.invoice_date.isoformat() if self.invoice_date else None,
            "total_ht": float(self.total_ht) if self.total_ht is not None else None,
            "vat": float(self.vat_amount) if self.vat_amount is not None else None,
            "stamp_amount": float(self.stamp_amount) if self.stamp_amount is not None else None,
            "total_ttc": float(self.total_ttc) if self.total_ttc is not None else None,
            "currency": self.currency,
            "document_type": self.document_type,
            "category": self.category,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "validated_at": self.validated_at.isoformat() if self.validated_at else None,
        }


class InvoiceFieldValue(db.Model):
    __tablename__ = "invoice_field_values"

    id = db.Column(db.Integer, primary_key=True)
    invoice_id = db.Column(db.Integer, db.ForeignKey("invoices.id"), nullable=False, index=True)
    field_name = db.Column(db.String(100), nullable=False)
    extracted_value = db.Column(db.Text, nullable=True)
    validated_value = db.Column(db.Text, nullable=True)
    confidence = db.Column(db.String(20), nullable=True)
    source = db.Column(db.String(30), nullable=False, default="ai")

    invoice = db.relationship("Invoice", back_populates="field_values")
    __table_args__ = (db.UniqueConstraint("invoice_id", "field_name", name="uq_invoice_field_name"),)


class ExportHistory(db.Model):
    __tablename__ = "export_histories"

    id = db.Column(db.Integer, primary_key=True)
    invoice_id = db.Column(db.Integer, db.ForeignKey("invoices.id"), nullable=False, index=True)
    export_format = db.Column(db.String(20), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    exported_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    invoice = db.relationship("Invoice", back_populates="exports")
