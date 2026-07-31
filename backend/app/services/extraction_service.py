from __future__ import annotations

import json
import os
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from flask import current_app

from app.models.invoice import ExtractionRun, InvoiceFieldValue, InvoiceStatus
from extensions import db

try:  # Pydantic validates Ollama's JSON in installed environments.
    from pydantic import BaseModel, ValidationError
except ImportError:  # Keep the local skeleton testable before dependencies are installed.
    BaseModel = None
    ValidationError = ValueError


AI_FIELDS = (
    "supplier", "tax_identifier", "invoice_number", "invoice_date", "total_ht",
    "vat_amount", "stamp_amount", "total_ttc", "currency", "category",
)
MONEY_FIELDS = {"total_ht", "vat_amount", "stamp_amount", "total_ttc"}


if BaseModel:
    class InvoiceAIResult(BaseModel):
        supplier: str | None = None
        tax_identifier: str | None = None
        invoice_number: str | None = None
        invoice_date: str | None = None
        total_ht: str | None = None
        vat_amount: str | None = None
        stamp_amount: str | None = None
        total_ttc: str | None = None
        currency: str | None = "TND"
        category: str | None = None
        confidence: dict[str, str] = {}
        missing_fields: list[str] = []
        warnings: list[str] = []
else:
    InvoiceAIResult = None


class ExtractionService:
    """Extract a document, ask Ollama for structured data, and save an auditable run."""

    def process_document(self, document):
        document.status = InvoiceStatus.PROCESSING.value
        document.invoice.status = InvoiceStatus.PROCESSING.value
        db.session.commit()
        run = ExtractionRun(document_id=document.id, ai_model=current_app.config["OLLAMA_MODEL"])
        db.session.add(run)

        try:
            text = self.extract_text(Path(document.storage_path))
            if not text.strip():
                raise ValueError("No readable text was found in this document")
            run.ocr_text = text
            result, model = self.extract_structured_data(text)
            result = self.validate_result(result)
            run.ai_model = model
            run.ai_raw_json = result
            run.success = True
            self.apply_result(document.invoice, result, source="ai" if model != "heuristic-fallback" else "heuristic")
            document.status = InvoiceStatus.NEEDS_REVIEW.value
            document.invoice.status = InvoiceStatus.NEEDS_REVIEW.value
            db.session.commit()
            return document.invoice, result
        except Exception as exc:
            run.success = False
            run.error_message = str(exc)
            document.status = InvoiceStatus.FAILED.value
            document.invoice.status = InvoiceStatus.FAILED.value
            db.session.commit()
            raise

    def extract_text(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            return self._extract_pdf_text(path)
        return self._ocr_image(path)

    def _extract_pdf_text(self, path: Path) -> str:
        try:
            import fitz  # PyMuPDF
            pdf = fitz.open(path)
            text = "\n".join(page.get_text("text") for page in pdf).strip()
            if text:
                return text
            images = [page.get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png") for page in pdf]
            return "\n".join(self._ocr_image_bytes(image) for image in images)
        except ImportError as exc:
            raise RuntimeError("PyMuPDF must be installed for PDF extraction") from exc

    def _ocr_image(self, path: Path) -> str:
        return self._ocr_image_bytes(path.read_bytes())

    def _ocr_image_bytes(self, payload: bytes) -> str:
        try:
            import cv2
            import numpy as np
            import pytesseract
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Invalid image")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            requested = current_app.config["TESSERACT_LANGUAGES"].split("+")
            tessdata_dir = Path(current_app.config["TESSDATA_DIR"])
            if tessdata_dir.exists():
                os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)
            available = set(pytesseract.get_languages())
            languages = [language for language in requested if language in available]
            if not languages:
                raise RuntimeError(
                    "No requested Tesseract language is installed. "
                    f"Expected one of: {', '.join(requested)}"
                )
            return pytesseract.image_to_string(gray, lang="+".join(languages))
        except ImportError as exc:
            raise RuntimeError("OpenCV and Tesseract dependencies are required for OCR") from exc

    def extract_structured_data(self, text: str) -> tuple[dict[str, Any], str]:
        prompt = """You extract invoice data. Return only one JSON object, without Markdown.
Use this exact schema: supplier, tax_identifier, invoice_number, invoice_date (YYYY-MM-DD),
total_ht, vat_amount, stamp_amount, total_ttc (decimal strings), currency, category,
confidence (object with field names and low/medium/high), missing_fields (array), warnings (array).
Use null when a value is absent. Do not invent values. Invoice text:\n""" + text[:18000]
        try:
            body = json.dumps({
                "model": current_app.config["OLLAMA_MODEL"], "prompt": prompt, "format": "json", "stream": False,
            }).encode("utf-8")
            request = Request(
                f"{current_app.config['OLLAMA_URL'].rstrip('/')}/api/generate",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=90) as response:
                raw = json.loads(response.read().decode("utf-8")).get("response", "")
            return json.loads(raw), current_app.config["OLLAMA_MODEL"]
        except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            # The project remains usable without a local model; the user sees a warning and reviews it.
            result = self._heuristic_extract(text)
            result["warnings"].append(f"Ollama unavailable or invalid JSON: {exc}")
            return result, "heuristic-fallback"

    def _heuristic_extract(self, text: str) -> dict[str, Any]:
        amounts = re.findall(r"(?:total\s*(?:ttc|ht)|tva|vat)\D{0,12}([0-9][0-9 .,'\u00a0]*)", text, re.I)
        number = re.search(r"(?:facture|invoice)\s*(?:n[°o]|number|no)?\s*[:#-]?\s*([A-Z0-9][A-Z0-9/_-]{2,})", text, re.I)
        date_match = re.search(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", text)
        currency = re.search(r"\b(TND|EUR|USD|DT)\b", text, re.I)
        values = [self._decimal_string(value) for value in amounts]
        return {
            "supplier": next((line.strip() for line in text.splitlines() if len(line.strip()) > 3), None),
            "tax_identifier": None,
            "invoice_number": number.group(1) if number else None,
            "invoice_date": self._normalise_date(date_match.group(1)) if date_match else None,
            "total_ht": values[0] if len(values) > 0 else None,
            "vat_amount": values[1] if len(values) > 1 else None,
            "stamp_amount": None,
            "total_ttc": values[-1] if values else None,
            "currency": currency.group(1).upper().replace("DT", "TND") if currency else "TND",
            "category": None,
            "confidence": {},
            "missing_fields": [],
            "warnings": [],
        }

    @staticmethod
    def _decimal_string(value: str | None) -> str | None:
        if not value:
            return None
        normalized = value.replace("\u00a0", "").replace(" ", "").replace("'", "")
        if normalized.count(",") == 1 and normalized.count(".") == 0:
            normalized = normalized.replace(",", ".")
        elif normalized.count(",") and normalized.count("."):
            normalized = normalized.replace(",", "")
        try:
            return f"{Decimal(normalized):.3f}"
        except InvalidOperation:
            return None

    @staticmethod
    def _normalise_date(value: str) -> str | None:
        parts = re.split("[-/]", value)
        try:
            if len(parts[0]) == 4:
                return date(int(parts[0]), int(parts[1]), int(parts[2])).isoformat()
            year = int(parts[2]) + (2000 if len(parts[2]) == 2 else 0)
            return date(year, int(parts[1]), int(parts[0])).isoformat()
        except ValueError:
            return None

    def validate_result(self, result: Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise ValueError("The AI response must be a JSON object")
        if InvoiceAIResult:
            try:
                result = InvoiceAIResult.model_validate(result).model_dump(mode="json")
            except ValidationError as exc:
                raise ValueError(f"AI JSON does not match the expected schema: {exc}") from exc
        clean: dict[str, Any] = {field: result.get(field) for field in AI_FIELDS}
        for field in MONEY_FIELDS:
            clean[field] = self._decimal_string(clean[field])
        if clean["invoice_date"]:
            try:
                clean["invoice_date"] = date.fromisoformat(str(clean["invoice_date"])).isoformat()
            except ValueError:
                clean["invoice_date"] = None
        clean["currency"] = str(clean["currency"] or "TND").upper()[:10]
        clean["confidence"] = result.get("confidence") if isinstance(result.get("confidence"), dict) else {}
        clean["missing_fields"] = result.get("missing_fields") if isinstance(result.get("missing_fields"), list) else []
        clean["warnings"] = result.get("warnings") if isinstance(result.get("warnings"), list) else []
        clean["missing_fields"] = sorted(set(clean["missing_fields"] + [key for key in AI_FIELDS if not clean.get(key)]))
        return clean

    def apply_result(self, invoice, result: dict[str, Any], source: str):
        confidence = result.get("confidence", {})
        for field in AI_FIELDS:
            value = result.get(field)
            if field in MONEY_FIELDS and value is not None:
                value = Decimal(value)
            elif field == "invoice_date" and value:
                value = date.fromisoformat(value)
            setattr(invoice, field, value)
            field_value = InvoiceFieldValue.query.filter_by(invoice_id=invoice.id, field_name=field).first()
            if not field_value:
                field_value = InvoiceFieldValue(invoice_id=invoice.id, field_name=field)
                db.session.add(field_value)
            field_value.extracted_value = str(value) if value is not None else None
            field_value.confidence = str(confidence.get(field, "low"))
            field_value.source = source
