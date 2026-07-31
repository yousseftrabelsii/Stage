import io
import os
import shutil
import tempfile

import pytest

from app import create_app
from config import Config
from extensions import db


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    UPLOAD_FOLDER = tempfile.mkdtemp(prefix="uploads-", dir=".")
    EXPORT_FOLDER = tempfile.mkdtemp(prefix="exports-", dir=".")
    AUTO_CREATE_DATABASE = True


@pytest.fixture()
def client():
    app = create_app(TestingConfig)
    app.config.update(TESTING=True)

    with app.app_context():
        db.drop_all()
        db.create_all()

    with app.test_client() as test_client:
        yield test_client

    with app.app_context():
        db.session.remove()
        db.drop_all()
    
    # Cleanup temporary folders
    shutil.rmtree(TestingConfig.UPLOAD_FOLDER, ignore_errors=True)
    shutil.rmtree(TestingConfig.EXPORT_FOLDER, ignore_errors=True)


def test_upload_creates_invoice_and_persists_file(client):
    response = client.post(
        "/api/uploads",
        data={"file": (io.BytesIO(b"%PDF-1.4 test"), "sample.pdf")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["status"] == "uploaded"
    assert payload["invoice_id"] is not None
    assert payload["document_id"] is not None
    invoices_response = client.get("/api/invoices")
    assert invoices_response.status_code == 200
    invoices_payload = invoices_response.get_json()
    assert len(invoices_payload["invoices"]) == 1
    assert invoices_payload["invoices"][0]["invoice_number"] is None


def test_validation_then_csv_export(client):
    upload = client.post(
        "/api/uploads",
        data={"file": (io.BytesIO(b"%PDF-1.4 test"), "sample.pdf")},
        content_type="multipart/form-data",
    ).get_json()
    validation = client.put(
        f"/api/invoices/{upload['invoice_id']}",
        json={"supplier": "STEG", "invoice_number": "76241339", "invoice_date": "2022-10-06",
              "total_ttc": "91.816", "currency": "TND", "category": "Energie"},
    )
    assert validation.status_code == 200
    assert validation.get_json()["invoice"]["status"] == "validated"
    exported = client.get("/api/exports?format=csv")
    assert exported.status_code == 200
    assert exported.headers["Content-Type"].startswith("text/csv")


def test_processing_creates_a_reviewable_ai_result(client, monkeypatch):
    from app.services.extraction_service import ExtractionService

    monkeypatch.setattr(ExtractionService, "extract_text", lambda *_: "Facture STEG 76241339")
    monkeypatch.setattr(
        ExtractionService,
        "extract_structured_data",
        lambda *_: ({
            "supplier": "STEG", "tax_identifier": None, "invoice_number": "76241339",
            "invoice_date": "2022-10-06", "total_ht": "76.000", "vat_amount": "15.816",
            "stamp_amount": None, "total_ttc": "91.816", "currency": "TND", "category": "Energie",
            "confidence": {"supplier": "high"}, "missing_fields": [], "warnings": [],
        }, "qwen2.5:3b"),
    )
    upload = client.post(
        "/api/uploads",
        data={"file": (io.BytesIO(b"%PDF-1.4 test"), "sample.pdf")},
        content_type="multipart/form-data",
    ).get_json()
    processed = client.post(f"/api/documents/{upload['document_id']}/process")
    assert processed.status_code == 200
    invoice = processed.get_json()["invoice"]
    assert invoice["status"] == "needs_review"
    assert invoice["total_ttc"] == "91.816"
