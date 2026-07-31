"""Audit tests: edge cases, accounting rules, API defects."""
from __future__ import annotations

import csv
import io
import json
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

from app import create_app
from app.models.invoice import InvoiceStatus
from app.services.extraction_service import ExtractionService
from config import Config
from extensions import db


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    UPLOAD_FOLDER = tempfile.mkdtemp(prefix="uploads-audit-")
    EXPORT_FOLDER = tempfile.mkdtemp(prefix="exports-audit-")
    AUTO_CREATE_DATABASE = True


@pytest.fixture()
def client():
    app = create_app(TestingConfig)
    with app.app_context():
        db.drop_all()
        db.create_all()
    with app.test_client() as test_client:
        yield test_client
    with app.app_context():
        db.session.remove()
        db.drop_all()


def _upload(client, name="facture.pdf", content=b"%PDF-1.4 sample"):
    return client.post(
        "/api/uploads",
        data={"file": (io.BytesIO(content), name)},
        content_type="multipart/form-data",
    )


def test_validation_accepts_empty_critical_fields(client):
    """Comptable: on peut valider une facture sans montants ni fournisseur."""
    inv_id = _upload(client).get_json()["invoice_id"]
    resp = client.put(f"/api/invoices/{inv_id}", json={"supplier": "", "total_ttc": None})
    assert resp.status_code == 200
    assert resp.get_json()["invoice"]["status"] == "validated"


def test_no_accounting_coherence_check(client):
    """Comptable: HT + TVA + timbre != TTC accepté sans alerte."""
    inv_id = _upload(client).get_json()["invoice_id"]
    resp = client.put(
        f"/api/invoices/{inv_id}",
        json={
            "supplier": "STEG",
            "total_ht": "100.000",
            "vat_amount": "19.000",
            "stamp_amount": "1.000",
            "total_ttc": "50.000",
            "currency": "TND",
        },
    )
    assert resp.status_code == 200
    invoice = resp.get_json()["invoice"]
    assert invoice["status"] == "validated"


def test_export_validation_status_becomes_exported_not_validated(client):
    """Export: colonne validation_status = 'exported' au lieu de 'validated'."""
    inv_id = _upload(client).get_json()["invoice_id"]
    client.put(
        f"/api/invoices/{inv_id}",
        json={"supplier": "STEG", "total_ttc": "91.816", "currency": "TND"},
    )
    exported = client.get("/api/exports?format=csv")
    assert exported.status_code == 200
    rows = list(csv.DictReader(io.StringIO(exported.data.decode("utf-8-sig"))))
    assert rows[0]["validation_status"] == "exported"


def test_reexport_same_invoices_again(client):
    """Export: les factures déjà exportées sont réexportées sans filtre."""
    inv_id = _upload(client).get_json()["invoice_id"]
    client.put(f"/api/invoices/{inv_id}", json={"supplier": "A", "total_ttc": "10.000"})
    assert client.get("/api/exports?format=csv").status_code == 200
    assert client.get("/api/exports?format=csv").status_code == 200


def test_duplicate_upload_same_hash_not_blocked(client):
    """Dev: même fichier importé deux fois -> doublons."""
    content = b"%PDF-1.4 duplicate test content"
    _upload(client, content=content)
    _upload(client, content=content)
    invoices = client.get("/api/invoices").get_json()["invoices"]
    assert len(invoices) == 2


def test_unsupported_extension_rejected(client):
    resp = client.post(
        "/api/uploads",
        data={"file": (io.BytesIO(b"hello"), "virus.exe")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_process_invalid_document_id(client):
    resp = client.post("/api/documents/99999/process")
    assert resp.status_code == 404


def test_decimal_string_parsing_ambiguous_formats():
    svc = ExtractionService()
    assert svc._decimal_string("1.234,56") == "1234.560"  # format EU?
    assert svc._decimal_string("1,234.56") == "1234.560"


def test_heuristic_date_dd_mm_yyyy():
    svc = ExtractionService()
    assert svc._normalise_date("06/10/2022") == "2022-10-06"


def test_validate_result_rejects_non_dict():
    svc = ExtractionService()
    with pytest.raises(ValueError, match="JSON object"):
        svc.validate_result("not json")


def test_ai_schema_missing_vat_field_in_old_schema_file():
    """Schema marshmallow obsolète: tax_amount au lieu de vat_amount."""
    from app.schemas.invoice_schema import InvoiceSchema

    fields = InvoiceSchema().fields
    assert "tax_amount" in fields
    assert "vat_amount" not in fields
    assert "stamp_amount" not in fields
    assert "tax_identifier" not in fields


def test_export_xlsx_format(client):
    inv_id = _upload(client).get_json()["invoice_id"]
    client.put(f"/api/invoices/{inv_id}", json={"supplier": "X", "total_ttc": "1.000"})
    resp = client.get("/api/exports?format=xlsx")
    assert resp.status_code == 200
    assert "spreadsheetml" in resp.headers["Content-Type"]


def test_export_no_invoices_returns_400(client):
    assert client.get("/api/exports?format=csv").status_code == 400


def test_invalid_export_format(client):
    inv_id = _upload(client).get_json()["invoice_id"]
    client.put(f"/api/invoices/{inv_id}", json={"supplier": "X", "total_ttc": "1.000"})
    assert client.get("/api/exports?format=pdf").status_code == 400
