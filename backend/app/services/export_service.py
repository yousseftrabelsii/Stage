from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv

from flask import current_app

from app.models.invoice import ExportHistory, InvoiceStatus
from extensions import db


EXPORT_COLUMNS = [
    "document_id", "filename", "supplier", "tax_identifier", "invoice_number", "invoice_date",
    "total_ht", "vat_amount", "stamp_amount", "total_ttc", "currency", "category",
    "validation_status", "validated_at",
]


class ExportService:
    def export_invoices(self, invoices, export_format: str) -> Path:
        if export_format not in {"csv", "xlsx"}:
            raise ValueError("Export format must be csv or xlsx")
        rows = [self._row(invoice) for invoice in invoices]
        folder = Path(current_app.config["EXPORT_FOLDER"])
        folder.mkdir(parents=True, exist_ok=True)
        filename = f"factures_{datetime.utcnow():%Y%m%d_%H%M%S_%f}.{export_format}"
        path = folder / filename
        if export_format == "csv":
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
        else:
            import pandas as pd
            frame = pd.DataFrame(rows, columns=EXPORT_COLUMNS)
            frame.to_excel(path, index=False, engine="openpyxl")

        for invoice in invoices:
            db.session.add(ExportHistory(invoice_id=invoice.id, export_format=export_format, filename=filename, file_path=str(path)))
            invoice.status = InvoiceStatus.EXPORTED.value
        db.session.commit()
        return path

    @staticmethod
    def _row(invoice):
        return {
            "document_id": invoice.document_id,
            "filename": invoice.document.original_name,
            "supplier": invoice.supplier,
            "tax_identifier": invoice.tax_identifier,
            "invoice_number": invoice.invoice_number,
            "invoice_date": invoice.invoice_date.isoformat() if invoice.invoice_date else None,
            "total_ht": invoice.total_ht,
            "vat_amount": invoice.vat_amount,
            "stamp_amount": invoice.stamp_amount,
            "total_ttc": invoice.total_ttc,
            "currency": invoice.currency,
            "category": invoice.category,
            "validation_status": invoice.status,
            "validated_at": invoice.validated_at.isoformat() if invoice.validated_at else None,
        }
