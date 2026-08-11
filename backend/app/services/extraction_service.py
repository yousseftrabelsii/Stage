"""
Extraction service — merged OCR + AI pipeline.

Priority order for text extraction:
  PDF  → PyMuPDF (text layer) → pdfplumber → page-by-page OCR
  Image → OpenCV preprocessing → PaddleOCR (multi-lang, angle-corrected)

Priority order for structured extraction:
  Ollama / Qwen2.5 (ollama.chat, French system prompt) → spaCy NER fallback → heuristic regex fallback
"""
from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from flask import current_app

from app.models.invoice import ExtractionRun, InvoiceFieldValue, InvoiceStatus
from extensions import db

try:
    from pydantic import BaseModel, ValidationError
except ImportError:
    BaseModel = None  # type: ignore[assignment,misc]
    ValidationError = ValueError  # type: ignore[assignment,misc]


# ── Field definitions ──────────────────────────────────────────────────────────

AI_FIELDS = (
    "supplier",
    "tax_identifier",
    "invoice_number",
    "invoice_date",
    "total_ht",
    "vat_amount",
    "stamp_amount",
    "total_ttc",
    "currency",
    "document_type",
    "category",
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
        document_type: str | None = "Facture"
        category: str | None = None
        confidence: dict[str, str] = {}
        missing_fields: list[str] = []
        warnings: list[str] = []
else:
    InvoiceAIResult = None  # type: ignore[assignment,misc]


# ── spaCy (lazy-loaded) ────────────────────────────────────────────────────────

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("fr_core_news_sm")
        except Exception:
            try:
                import spacy
                _nlp = spacy.blank("fr")
            except ImportError:
                _nlp = None
    return _nlp


# ── PaddleOCR lazy singleton ───────────────────────────────────────────────────

_paddle_ocr = None


def _get_paddle_ocr():
    """Return a cached PaddleOCR instance (loads models only once)."""
    global _paddle_ocr
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR
            # lang='fr' enables French + English simultaneously;
            # use_angle_cls=True corrects rotated text blocks.
            _paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang="fr",
                show_log=False,
            )
            print("[PaddleOCR] Model loaded successfully.")
        except Exception as exc:
            print(f"[PaddleOCR] Failed to initialise: {exc}")
            _paddle_ocr = None
    return _paddle_ocr


# ═══════════════════════════════════════════════════════════════════════════════
# ExtractionService
# ═══════════════════════════════════════════════════════════════════════════════

class ExtractionService:
    """Extract a document, ask Ollama for structured data, and save an auditable run."""

    # ── Public entry point ─────────────────────────────────────────────────────

    def process_document(self, document):
        """OCR → AI extraction → persist to Invoice. Returns (invoice, result_dict)."""
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

            result, model_used = self.extract_structured_data(text)
            result = self.validate_result(result, text)

            run.ai_model = model_used
            run.ai_raw_json = result
            run.success = True

            source = "ai" if model_used not in ("heuristic-fallback", "spacy-fallback") else model_used
            self.apply_result(document.invoice, result, source=source)

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

    # ── Text extraction ────────────────────────────────────────────────────────

    def extract_text(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            return self._extract_pdf_text(path)
        return self._ocr_image(path)

    def _extract_pdf_text(self, path: Path) -> str:
        text = ""

        # Primary: PyMuPDF — fast, best for text-based PDFs
        try:
            import fitz
            doc = fitz.open(str(path))
            for page in doc:
                text += page.get_text()
            doc.close()
            text = text.strip()
        except Exception as exc:
            print(f"[PyMuPDF] Extraction failed: {exc}")
            text = ""

        # Fallback 1: pdfplumber — better at structured/tabular PDFs + table extraction
        if not text:
            try:
                import pdfplumber
                with pdfplumber.open(str(path)) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
                        # Also extract tables — preserves amounts/labels that flow in columns
                        for table in page.extract_tables():
                            for row in table:
                                if row:
                                    text += " | ".join(cell or "" for cell in row) + "\n"
                text = text.strip()
            except Exception as exc:
                print(f"[pdfplumber] Extraction failed: {exc}")

        # Fallback 2: page-by-page OCR (scanned PDF)
        if not text:
            try:
                import fitz
                doc = fitz.open(str(path))
                for page_num in range(len(doc)):
                    # 300 DPI equivalent (factor 300/72 ≈ 4.17)
                    pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(4, 4))
                    img_bytes = pix.tobytes("png")
                    text += self._ocr_image_bytes(img_bytes) + "\n"
                doc.close()
                text = text.strip()
            except Exception as exc:
                print(f"[Scanned PDF OCR] Failed: {exc}")

        return text

    def _ocr_image(self, path: Path) -> str:
        return self._ocr_image_bytes(path.read_bytes())

    def _ocr_image_bytes(self, payload: bytes) -> str:
        """PaddleOCR with enhanced OpenCV preprocessing."""
        try:
            import cv2
            import numpy as np

            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Invalid image data")

            # ── Upscale to at least 2400px on the long side for 300 DPI quality ──
            h, w = image.shape[:2]
            long_side = max(h, w)
            if long_side < 2400:
                scale = 2400 / long_side
                image = cv2.resize(image, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_CUBIC)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # ── Denoise ──
            gray = cv2.fastNlMeansDenoising(gray, h=10)

            # ── CLAHE — improve local contrast for faded/uneven scans ──
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # ── Sharpen — improves blurry/photocopied documents ──
            kernel_sharpen = np.array([[-1, -1, -1],
                                        [-1,  9, -1],
                                        [-1, -1, -1]])
            gray = cv2.filter2D(gray, -1, kernel_sharpen)

            # ── Deskew ── (straighten rotated scans up to ±10°)
            # Work on a binary copy for deskew detection, then apply to color image
            _, binary_deskew = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            binary_deskew = self._deskew(binary_deskew)

            # Reconstruct a color (BGR) enhanced image from deskewed gray for PaddleOCR
            enhanced_gray = self._deskew(gray)
            enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

            # ── Run PaddleOCR ──
            ocr = _get_paddle_ocr()
            if ocr is None:
                raise RuntimeError("PaddleOCR could not be initialised")

            result = ocr.ocr(enhanced_bgr, cls=True)

            # ── Flatten results into text lines ──
            lines: list[str] = []
            if result:
                for page in result:
                    if page is None:
                        continue
                    for box in page:
                        # box = [[coords], (text, confidence)]
                        text_info = box[1]
                        text_line = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
                        text_line = text_line.strip()
                        if text_line:
                            lines.append(text_line)

            recognized_text = "\n".join(lines).strip()
            print(f"[PaddleOCR] Extracted {len(lines)} lines.")
            return recognized_text

        except ImportError as exc:
            raise RuntimeError("OpenCV and PaddleOCR are required for OCR") from exc
        except Exception as exc:
            import traceback
            print(f"[PaddleOCR] OCR failed: {exc}")
            traceback.print_exc()
            return ""

    @staticmethod
    def _deskew(image):
        """Rotate the image to correct skew using Hough line analysis."""
        try:
            import cv2
            import numpy as np
            coords = np.column_stack(np.where(image < 128))  # dark pixels
            if len(coords) < 100:
                return image
            angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) < 0.5:   # negligible — skip rotation
                return image
            (h, w) = image.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            return cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            return image

    # ── Structured data extraction ─────────────────────────────────────────────

    def _guess_category_from_text(self, text: str) -> str | None:
        """Keyword-based category detection from OCR text."""
        text_lower = text.lower()
        keywords: dict[str, list[str]] = {
            "Achats de marchandises": ["marchandise", "revente", "grossiste", "stock", "article"],
            "Matières premières":     ["matière première", "bois", "acier", "farine", "ciment", "fer"],
            "Fournitures de bureau":  ["fourniture", "papier", "stylo", "cartouche", "imprimante", "papeterie", "encre"],
            "Informatique":           ["ordinateur", "logiciel", "licence", "pc", "serveur", "clavier", "écran", "informatique", "software", "hardware"],
            "Télécommunication":      ["télécom", "téléphone", "internet", "fibre", "mobile", "forfait", "adsl", "ooredoo", "orange", "telecom", "abonnement", "tunisie telecom"],
            "Eau, électricité, gaz":  ["eau", "électricité", "gaz", "steg", "sonede", "énergie"],
            "Loyer":                  ["loyer", "magasin", "entrepôt", "location", "bail", "agence immobilière"],
            "Assurance":              ["assurance", "responsabilité civile", "prime", "sinistre", "mutuelle", "star", "gat", "comar", "ami"],
            "Publicité et marketing": ["publicité", "marketing", "facebook ads", "google ads", "affiche", "campagne", "promotion", "sponsor", "flyer"],
            "Transport":              ["transport", "taxi", "carburant", "livraison", "essence", "gasoil", "péage", "fret", "logistique", "agil", "total"],
            "Déplacements":           ["déplacement", "billet d'avion", "hôtel", "mission", "voyage", "hébergement", "train", "tunisair"],
            "Entretien et réparation":["entretien", "réparation", "maintenance", "dépannage", "pièce de rechange", "garage", "vidange"],
            "Honoraires":             ["honoraire", "comptable", "avocat", "consultant", "notaire", "expert", "conseil", "audit"],
            "Formation":              ["formation", "cours", "séminaire", "apprentissage", "coaching"],
            "Frais bancaires":        ["frais bancaire", "commission", "tenue de compte", "agios", "biat", "amen", "atb"],
            "Impôts et taxes":        ["impôt", "taxe", "douane", "timbre", "fiscal", "retenue", "recette des finances"],
            "Salaires et charges sociales": ["salaire", "rémunération", "personnel", "paie", "fiche de paie", "cnss"],
            "Immobilisations":        ["machine", "mobilier", "bâtiment", "équipement", "investissement"],
        }
        for category, words in keywords.items():
            for word in words:
                if word in text_lower:
                    return category
        return None

    def _get_best_model(self) -> str:
        """Return the best available Ollama model, preferring Qwen2.5 variants."""
        try:
            import ollama as ollama_lib
            models_info = ollama_lib.list()
            models = []
            if hasattr(models_info, 'models'):
                models = [m.model for m in models_info.models]
            elif isinstance(models_info, dict) and 'models' in models_info:
                models = [m.get('model') if isinstance(m, dict) else getattr(m, 'model', None) for m in models_info['models']]
                models = [m for m in models if m]

            if not models:
                return current_app.config.get("OLLAMA_MODEL", "qwen2.5:7b")

            # Priority: Qwen2.5 (best accuracy, fits 8 GB RAM with Q4 quantisation)
            # then older Qwen, then Llama3 variants as fallback.
            for preferred in [
                'qwen2.5:7b', 'qwen2.5:3b', 'qwen2.5',
                'qwen2:7b', 'qwen2', 'qwen',
                'llama3.1', 'llama3:latest', 'llama3', 'mistral', 'llama',
            ]:
                if preferred in models:
                    return preferred
                for m in models:
                    if preferred in m.lower():
                        return m
            return models[0]
        except Exception as e:
            print(f"[Ollama] Failed to detect models: {e}")
            return current_app.config.get("OLLAMA_MODEL", "qwen2.5:7b")

    def extract_structured_data(self, text: str) -> tuple[dict[str, Any], str]:
        """Try Ollama first, fall back to spaCy NER, then regex heuristics."""

        # ── Ollama (primary) ──────────────────────────────────────────────────
        try:
            import ollama as ollama_lib

            system_prompt = (
                "Tu es un expert comptable tunisien spécialisé dans l'extraction précise de données de factures.\n"
                "Le texte fourni provient d'un OCR multi-passes : il peut contenir des doublons, artefacts et bruit.\n\n"
                "RÈGLES ABSOLUES — respecte-les sans exception :\n"
                "R1. NE JAMAIS calculer la TVA mathématiquement. Si vat_amount n'est pas écrit en toutes lettres dans le document, retourne null.\n"
                "R2. FOURNISSEUR = l'ÉMETTEUR de la facture (généralement en grand en haut de page, ou section 'Fournisseur/Exporter'). "
                "Ce n'est pas le client/destinataire. Exclus les mentions génériques (ex: 'Facture', 'Client', 'Doit'). Prends la raison sociale exacte.\n"
                "R3. DATE = date d'émission/facture uniquement. Ignore toute date d'échéance ou de livraison.\n"
                "R4. MONTANTS tunisiens : '3,732' = 3.732 DT (virgule = décimal). '1,400.00' = 1400.000 (virgule = milliers). "
                "Retourne toujours un nombre décimal en string (ex: '3.732', '1400.000').\n"
                "R5. ARTEFACTS OCR : corrige '0'/'O', '1'/'l'/'I', espaces dans les chiffres, tirets parasites. "
                "Si un montant contient des espaces comme '3 732', c'est 3732.\n"
                "R6. MULTI-PASSES : si un champ apparaît dans plusieurs passes OCR avec des valeurs différentes, "
                "prends la valeur la plus longue et la plus cohérente avec le contexte.\n"
                "R7. NUMÉRO DE FACTURE : inclure les préfixes/suffixes (ex: 'FAC-2024-001', 'F/2024/123'). Ne pas tronquer.\n"
                "R8. MATRICULE FISCAL tunisien : format typique '1234567A/B/C/000' ou '1234567X/A/P/000'. Retourne-le tel quel.\n"
                "R9. document_type : exactement l'une de ces valeurs : 'Facture', 'Reçu', 'Avoir', 'Note', 'Autre'.\n"
                "R10. Retourne UNIQUEMENT un JSON valide, sans markdown, sans explication, sans texte avant ou après.\n"
                "VÉRIFICATION : avant de répondre, vérifie que total_ttc ≈ total_ht + vat_amount + stamp_amount. "
                "Si l'écart est > 1%, marque une warning. Ne corrige pas les valeurs, signale seulement."
            )

            user_prompt = (
                "Extrait les informations suivantes du texte OCR ci-dessous.\n"
                "Retourne STRICTEMENT ce schéma JSON (toutes les clés, null si non trouvé) :\n"
                "{\n"
                '  "supplier": "Nom exact du fournisseur émetteur",\n'
                '  "tax_identifier": "Matricule fiscal ou null",\n'
                '  "invoice_number": "Numéro complet de la facture ou null",\n'
                '  "invoice_date": "YYYY-MM-DD ou null",\n'
                '  "total_ht": "montant en string ex: 1500.000 ou null",\n'
                '  "vat_amount": "montant TVA en string ou null (jamais calculé)",\n'
                '  "stamp_amount": "montant timbre en string ou null",\n'
                '  "total_ttc": "montant TTC en string ou null",\n'
                '  "currency": "TND par défaut, sinon EUR/USD/etc.",\n'
                '  "document_type": "Facture|Reçu|Avoir|Note|Autre",\n'
                '  "category": "catégorie parmi la liste ou null",\n'
                '  "confidence": {"supplier":"high","invoice_number":"medium",...},\n'
                '  "missing_fields": ["champs absents"],\n'
                '  "warnings": ["anomalies détectées"]\n'
                "}\n\n"
                "Catégories autorisées UNIQUEMENT : "
                "Achats de marchandises, Matières premières, Fournitures de bureau, Informatique, "
                "Télécommunication, Eau électricité gaz, Loyer, Assurance, Publicité et marketing, "
                "Transport, Déplacements, Entretien et réparation, Honoraires, Formation, "
                "Frais bancaires, Impôts et taxes, Salaires et charges sociales, Immobilisations.\n\n"
                f"=== TEXTE OCR ===\n{text[:12000]}\n=== FIN DU TEXTE ===\n\nJSON:"
            )



            model_name = self._get_best_model()

            # Wrap in a thread so we can enforce a hard timeout if Ollama hangs
            import threading
            _result_box: list = []
            _err_box:    list = []

            def _call():
                try:
                    resp = ollama_lib.chat(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_prompt},
                        ],
                        format="json",
                        options={"temperature": 0.0},
                    )
                    _result_box.append(resp)
                except Exception as e:
                    _err_box.append(e)

            t = threading.Thread(target=_call, daemon=True)
            t.start()
            t.join(timeout=60)   # give model up to 60 s to respond

            if _err_box:
                raise _err_box[0]
            if not _result_box:
                raise TimeoutError("Ollama did not respond within 60 seconds")

            response = _result_box[0]
            content = response.message.content.strip()
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                content = match.group(0)
            return json.loads(content), model_name

        except Exception as exc:
            print(f"[Ollama] Failed ({exc}), trying spaCy fallback…")

        # ── spaCy NER (secondary fallback) ───────────────────────────────────
        nlp = _get_nlp()
        if nlp is not None:
            try:
                result = self._spacy_extract(text, nlp)
                result["warnings"] = ["Ollama unavailable; extracted with spaCy NER"]
                return result, "spacy-fallback"
            except Exception as exc:
                print(f"[spaCy] Failed ({exc}), using heuristic fallback…")

        # ── Regex heuristics (last resort) ───────────────────────────────────
        result = self._heuristic_extract(text)
        result["warnings"] = ["Ollama and spaCy unavailable; extracted with heuristics"]
        return result, "heuristic-fallback"

    # ── Fallback extractors ────────────────────────────────────────────────────

    def _spacy_extract(self, text: str, nlp) -> dict[str, Any]:
        """spaCy NER-based extraction with heuristic helpers."""
        doc = nlp(text)
        supplier = None
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PER"):
                supplier = ent.text
                break
        if not supplier:
            supplier = self._extract_supplier(text)

        currency_match = re.search(r"\b(TND|EUR|USD|DT|GBP|MAD)\b", text, re.I)
        currency = "TND"
        if currency_match:
            c = currency_match.group(1).upper()
            currency = "TND" if c == "DT" else c

        total_ttc = self._extract_labelled_amount(
            text, r"(?:montant\s+(?:total|ttc)|total\s+ttc|total\s+[àa]\s+payer|net\s+[àa]\s+payer)"
        )
        if not total_ttc:
            all_amounts = re.findall(r"\b(\d{1,3}(?:[,\s]\d{3})*[.,]\d{2,3})\b", text)
            parsed = [(float(self._decimal_string(a) or "0"), self._decimal_string(a))
                      for a in all_amounts if self._decimal_string(a)]
            total_ttc = max(parsed, key=lambda x: x[0])[1] if parsed else None

        return {
            "supplier":       supplier,
            "tax_identifier": self._extract_tax_id(text),
            "invoice_number": self._extract_invoice_number(text),
            "invoice_date":   self._extract_date(text),
            "total_ht":       self._extract_labelled_amount(text, r"(?:montant\s+ht|total\s+ht|ht\s+ap[rè]+s)"),
            "vat_amount":     self._extract_labelled_amount(text, r"(?:tva|vat|taxe)[\s\d%]*"),
            "stamp_amount":   self._extract_labelled_amount(text, r"(?:timbre|stamp)"),
            "total_ttc":      total_ttc,
            "currency":       currency,
            "document_type":  "Facture",
            "category":       self._guess_category_from_text(text),
            "confidence":     {},
            "missing_fields": [],
            "warnings":       [],
        }

    def _heuristic_extract(self, text: str) -> dict[str, Any]:
        """Improved regex heuristic extraction tuned for Tunisian/French invoices."""
        supplier     = self._extract_supplier(text)
        tax_id       = self._extract_tax_id(text)
        inv_number   = self._extract_invoice_number(text)
        invoice_date = self._extract_date(text)
        total_ht     = self._extract_labelled_amount(text, r"(?:montant\s+ht|total\s+ht|ht\s+ap[rè]+s)")
        vat_amount   = self._extract_labelled_amount(text, r"(?:tva|vat|taxe)[\s\d%]*")
        stamp_amount = self._extract_labelled_amount(text, r"(?:timbre|stamp)")
        total_ttc    = self._extract_labelled_amount(
            text, r"(?:montant\s+(?:total|ttc)|total\s+ttc|total\s+[àa]\s+payer|net\s+[àa]\s+payer)"
        )
        if not total_ttc:
            all_amounts = re.findall(r"\b(\d{1,3}(?:[,\s]\d{3})*[.,]\d{2,3})\b", text)
            parsed = []
            for a in all_amounts:
                v = self._decimal_string(a)
                if v:
                    try:
                        parsed.append((float(v), v))
                    except ValueError:
                        pass
            if parsed:
                total_ttc = max(parsed, key=lambda x: x[0])[1]

        currency_match = re.search(r"\b(TND|EUR|USD|DT|GBP|MAD)\b", text, re.I)
        currency = "TND"
        if currency_match:
            c = currency_match.group(1).upper()
            currency = "TND" if c == "DT" else c

        return {
            "supplier":       supplier,
            "tax_identifier": tax_id,
            "invoice_number": inv_number,
            "invoice_date":   invoice_date,
            "total_ht":       total_ht,
            "vat_amount":     vat_amount,
            "stamp_amount":   stamp_amount,
            "total_ttc":      total_ttc,
            "currency":       currency,
            "document_type":  "Facture",
            "category":       self._guess_category_from_text(text),
            "confidence":     {},
            "missing_fields": [],
            "warnings":       [],
        }

    # ── Shared extraction helpers ──────────────────────────────────────────────

    _KNOWN_SUPPLIERS = [
        "Ooredoo", "Tunisie Telecom", "Orange Tunisie",
        "STEG", "Société Tunisienne de l'Electricité et du Gaz",
        "SONEDE",
        "TOPNET", "Hexabyte",
        "BIAT", "STB", "BNA", "Attijari", "Amen Bank", "UIB",
        "STAR", "GAT", "COMAR", "AMI",
        "AGIL", "TOTAL",
    ]

    def _extract_supplier(self, text: str) -> str | None:
        text_lower = text.lower()
        for brand in self._KNOWN_SUPPLIERS:
            if brand.lower() in text_lower:
                return brand
                
        forbidden_words = {"facture", "reçu", "bon de", "devis", "client", "doit", "code", "date", "page", "tél", "fax", "tel", "email", "mail", "adresse"}

        mf_block = re.search(
            r"(.{0,80})\n[^\n]*(?:MF|matricule fiscale?|identifiant fiscal)[^\n]*",
            text, re.I
        )
        if mf_block:
            lines = mf_block.group(1).strip().splitlines()
            for line in reversed(lines):
                candidate = line.strip()
                if 3 < len(candidate) < 80 and not any(w in candidate.lower() for w in forbidden_words):
                    return candidate
                    
        for line in text.splitlines():
            line = line.strip()
            if 4 <= len(line) <= 70 and re.search(r"[A-Za-zÀ-ÿ]{3}", line) and not re.fullmatch(r"[\d\s.,:;/\\-]+", line):
                if not any(w in line.lower() for w in forbidden_words):
                    return line
        return None

    def _extract_tax_id(self, text: str) -> str | None:
        # Handles Tunisian MF format: 789012H|M|A|000 (pipes allowed, RC suffix optional)
        m = re.search(
            r"(?:MF|matricule\s+fiscale?|identifiant\s+fiscal|n[°o]?\s*TVA)[^\n:]*[:\s]+([A-Z0-9/|\\]{5,30})",
            text, re.I
        )
        if m:
            return m.group(1).strip()
        # Fallback: bare Tunisian MF pattern (digits + letter + pipes)
        m2 = re.search(r"\b(\d{6,9}[A-Z](?:[|/][A-Z0-9]+){1,4})\b", text)
        return m2.group(1).strip() if m2 else None

    def _extract_invoice_number(self, text: str) -> str | None:
        patterns = [
            r"[Ff]acture\s+[Nn][°oO]?\.?\s*:?\s*([A-Z0-9][-A-Z0-9/]{2,})",
            r"[Ff]acture\s+[Nn][°oO]?\.?\s*:?\s*(\d{4,})",
            r"[Ii]nvoice\s+[Nn][uo]?\.?\s*:?\s*([A-Z0-9][-A-Z0-9/]{2,})",
            r"[Rr][éeÉE]f[eé]rence\s*[:\s]+([A-Z0-9][A-Z0-9/_-]{2,})",
            r"N[°oO]?\s*[Ff]acture\s*[:\s]+([A-Z0-9][-A-Z0-9/]{2,})",
            r"\bN[°oO]\s*(\d{6,})\b",
            r"(?:Num[eé]ro|N°|No\.?)\s*[:\s]*([A-Z0-9][-A-Z0-9/]{3,})",
            r"(?:Bon de commande|BC)\s*[Nn]?[°oO]?\.?\s*:?\s*([A-Z0-9][-A-Z0-9/]{2,})",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(1).strip()
        return None

    def _extract_date(self, text: str) -> str | None:
        # 1 — date near a label
        labelled = re.search(
            r"(?:date\s+(?:limite\s+de\s+)?(?:paiement|facture|[eé]mission|du?\s+document|d[ée]livrance)[^\n]{0,30})"
            r"(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}[/-]\d{2}[/-]\d{2})",
            text, re.I
        )
        target = labelled.group(1) if labelled else None

        # 2 — written month name (e.g. "15 janvier 2024" or "15/Jan/2024")
        if not target:
            month_map = {
                "janvier": "01", "février": "02", "mars": "03", "avril": "04",
                "mai": "05", "juin": "06", "juillet": "07", "août": "08",
                "septembre": "09", "octobre": "10", "novembre": "11", "décembre": "12",
                "jan": "01", "fev": "02", "mar": "03", "avr": "04",
                "jun": "06", "jul": "07", "aou": "08", "sep": "09",
                "oct": "10", "nov": "11", "dec": "12",
            }
            m = re.search(
                r"(\d{1,2})\s+("
                + "|".join(month_map.keys())
                + r")\s+(\d{2,4})",
                text, re.I
            )
            if m:
                day, mon, year = m.group(1), m.group(2).lower(), m.group(3)
                year_i = int(year) + (2000 if len(year) == 2 else 0)
                return f"{year_i:04d}-{month_map[mon]}-{int(day):02d}"

        # 3 — bare date pattern
        if not target:
            m = re.search(
                r"\b(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}[/-]\d{2}[/-]\d{2})\b", text)
            target = m.group(1) if m else None

        if not target:
            return None
        return self._normalise_date(target.replace(".", "/"))

    def _extract_labelled_amount(self, text: str, label_pattern: str) -> str | None:
        # Allow the amount to appear on the SAME line or the NEXT line after the label
        pattern = (
            r"(?i)" + label_pattern
            + r"[^\n\d]{0,40}((?:\d{1,3}[\s,])*\d{1,3}[,.]\d{2,3})"
        )
        m = re.search(pattern, text, re.I | re.DOTALL)
        if m:
            return self._decimal_string(m.group(1))
        # Second attempt: label on one line, amount on the very next line
        pattern2 = (
            r"(?i)" + label_pattern
            + r"[^\n]*\n\s*((?:\d{1,3}[\s,])*\d{1,3}[,.]\d{2,3})"
        )
        m2 = re.search(pattern2, text, re.I)
        return self._decimal_string(m2.group(1)) if m2 else None

    # ── Result validation & normalisation ─────────────────────────────────────

    def validate_result(self, result: Any, text: str = "") -> dict[str, Any]:
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

        raw_currency = clean.get("currency")
        clean["currency"] = str(raw_currency or "TND").strip().upper()[:10] or "TND"

        # Normalise document_type
        allowed_types = {"Facture", "Reçu", "Avoir", "Note", "Autre"}
        dt = clean.get("document_type")
        clean["document_type"] = dt if dt in allowed_types else "Facture"
        
        # Normalise category
        allowed_categories = {
            "Achats de marchandises", "Matières premières", "Fournitures de bureau",
            "Informatique", "Télécommunication", "Eau, électricité, gaz", "Loyer",
            "Assurance", "Publicité et marketing", "Transport", "Déplacements",
            "Entretien et réparation", "Honoraires", "Formation", "Frais bancaires",
            "Impôts et taxes", "Salaires et charges sociales", "Immobilisations"
        }
        c = clean.get("category")
        if not c or c not in allowed_categories:
            c = self._guess_category_from_text(text)
        clean["category"] = c if c in allowed_categories else None

        clean["confidence"] = result.get("confidence") if isinstance(result.get("confidence"), dict) else {}
        clean["missing_fields"] = result.get("missing_fields") if isinstance(result.get("missing_fields"), list) else []
        clean["warnings"] = result.get("warnings") if isinstance(result.get("warnings"), list) else []
        clean["missing_fields"] = sorted(
            set(clean["missing_fields"] + [k for k in AI_FIELDS if not clean.get(k)])
        )

        # ── Arithmetic cross-check: TTC should equal HT + VAT + stamp ──────────
        try:
            ttc   = Decimal(clean["total_ttc"])   if clean.get("total_ttc")    else None
            ht    = Decimal(clean["total_ht"])     if clean.get("total_ht")     else None
            vat   = Decimal(clean["vat_amount"])   if clean.get("vat_amount")   else Decimal("0")
            stamp = Decimal(clean["stamp_amount"]) if clean.get("stamp_amount") else Decimal("0")
            if ttc and ht:
                computed = ht + vat + stamp
                if ttc > 0 and abs(computed - ttc) / ttc > Decimal("0.01"):
                    clean["warnings"].append(
                        f"Incohérence montants: HT({ht})+TVA({vat})+Timbre({stamp})={computed} ≠ TTC({ttc})"
                    )
        except (InvalidOperation, TypeError):
            pass

        return clean


    # ── Apply result to Invoice ORM object ─────────────────────────────────────

    def apply_result(self, invoice, result: dict[str, Any], source: str):
        confidence = result.get("confidence", {})
        for field in AI_FIELDS:
            value = result.get(field)
            if field in MONEY_FIELDS and value is not None:
                try:
                    value = Decimal(str(value))
                except InvalidOperation:
                    value = None
            elif field == "invoice_date" and value:
                try:
                    value = date.fromisoformat(str(value))
                except ValueError:
                    value = None
            setattr(invoice, field, value)

            # Persist per-field audit trail
            field_value = InvoiceFieldValue.query.filter_by(
                invoice_id=invoice.id, field_name=field
            ).first()
            if not field_value:
                field_value = InvoiceFieldValue(invoice_id=invoice.id, field_name=field)
                db.session.add(field_value)
            field_value.extracted_value = str(value) if value is not None else None
            field_value.confidence = str(confidence.get(field, "low"))
            field_value.source = source

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _decimal_string(value: str | None) -> str | None:
        """Parse an amount string into a normalised decimal string (3 decimal places).

        Handles:
          - Tunisian  : '3,732'  → 3.732   (comma = decimal, ≤3 digits after)
          - European  : '1.400,00' → 1400.000
          - US/Anglo  : '1,400.00' → 1400.000
          - Spaced    : '3 732,500' → 3732.500
          - OCR noise : 'DT', 'TND', currency symbols stripped
        """
        if not value:
            return None
        # Strip currency labels and whitespace variants
        normalized = re.sub(r"[A-Za-z$€£]", "", str(value))
        normalized = (
            normalized
            .replace("\u00a0", "")   # non-breaking space
            .replace("\u202f", "")   # narrow no-break space
            .replace(" ", "")
            .replace("'", "")
            .strip()
        )
        if not normalized:
            return None

        comma_count = normalized.count(",")
        dot_count   = normalized.count(".")

        if comma_count == 0 and dot_count == 0:
            # Pure integer like '1400'
            pass
        elif comma_count == 1 and dot_count == 0:
            # Could be Tunisian decimal ('3,732') or thousands ('1,400')
            after_comma = normalized.split(",")[1]
            if len(after_comma) <= 3 and len(after_comma) != 3:
                # Short fraction → decimal separator
                normalized = normalized.replace(",", ".")
            elif len(after_comma) == 3:
                # Ambiguous: if value > 999, treat comma as thousands sep; else decimal
                try:
                    int_part = int(normalized.split(",")[0])
                except ValueError:
                    int_part = 0
                if int_part >= 10:
                    normalized = normalized.replace(",", "")   # thousands
                else:
                    normalized = normalized.replace(",", ".")  # decimal
            else:
                normalized = normalized.replace(",", "")       # thousands
        elif dot_count == 1 and comma_count == 0:
            pass  # already valid decimal
        elif comma_count >= 1 and dot_count >= 1:
            # e.g. '1.400,00' (European) or '1,400.00' (US)
            last_comma = normalized.rfind(",")
            last_dot   = normalized.rfind(".")
            if last_comma > last_dot:
                # Comma is decimal separator: remove dots, replace comma
                normalized = normalized.replace(".", "").replace(",", ".")
            else:
                # Dot is decimal separator: remove commas
                normalized = normalized.replace(",", "")
        else:
            normalized = normalized.replace(",", "")

        try:
            return f"{Decimal(normalized):.3f}"
        except InvalidOperation:
            return None


    @staticmethod
    def _normalise_date(value: str) -> str | None:
        parts = re.split(r"[-/]", value)
        try:
            if len(parts[0]) == 4:
                return date(int(parts[0]), int(parts[1]), int(parts[2])).isoformat()
            year = int(parts[2]) + (2000 if len(parts[2]) == 2 else 0)
            return date(year, int(parts[1]), int(parts[0])).isoformat()
        except (ValueError, IndexError):
            return None
