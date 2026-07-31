from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone, date
from typing import Optional

db = SQLAlchemy()

class Invoice(db.Model):  # type: ignore[name-defined]
    __tablename__ = 'invoices'

    id             = db.Column(db.Integer,     primary_key=True)
    filename       = db.Column(db.String(255), nullable=False)
    supplier       = db.Column(db.String(255), nullable=True)
    invoice_number = db.Column(db.String(100), nullable=True)
    invoice_date   = db.Column(db.Date,        nullable=True)
    total_ht       = db.Column(db.Float,       nullable=True)
    vat            = db.Column(db.Float,       nullable=True)
    total_ttc      = db.Column(db.Float,       nullable=True)
    currency       = db.Column(db.String(10),  nullable=True, default='EUR')
    document_type  = db.Column(db.String(100), nullable=True)
    status         = db.Column(db.String(50),  default='En attente')
    extracted_text = db.Column(db.Text,        nullable=True)
    created_at     = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def __init__(
        self,
        filename: str,
        supplier: Optional[str] = None,
        invoice_number: Optional[str] = None,
        invoice_date: Optional[date] = None,
        total_ht: Optional[float] = None,
        vat: Optional[float] = None,
        total_ttc: Optional[float] = None,
        currency: Optional[str] = 'EUR',
        document_type: Optional[str] = None,
        status: str = 'En attente',
        extracted_text: Optional[str] = None,
    ):
        self.filename = filename
        self.supplier = supplier
        self.invoice_number = invoice_number
        self.invoice_date = invoice_date
        self.total_ht = total_ht
        self.vat = vat
        self.total_ttc = total_ttc
        self.currency = currency
        self.document_type = document_type
        self.status = status
        self.extracted_text = extracted_text

    def to_dict(self) -> dict:
        return {
            'id':             self.id,
            'filename':       self.filename,
            'supplier':       self.supplier,
            'invoice_number': self.invoice_number,
            'invoice_date':   self.invoice_date.isoformat() if self.invoice_date else None,
            'total_ht':       self.total_ht,
            'vat':            self.vat,
            'total_ttc':      self.total_ttc,
            'currency':       self.currency,
            'document_type':  self.document_type,
            'status':         self.status,
            'created_at':     self.created_at.isoformat() if self.created_at else None,
        }
