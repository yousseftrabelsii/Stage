"""
Extraction service — merged OCR + AI pipeline.

Priority order for text extraction:
  PDF  → PyMuPDF (text layer) → pdfplumber → page-by-page OCR
  Image → OpenCV preprocessing → dual-pass Tesseract (PSM 3 + PSM 6)

Priority order for structured extraction:
  Ollama (ollama.chat, French system prompt) → spaCy NER fallback → heuristic regex fallback
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


# ── Tesseract binary configuration ────────────────────────────────────────────

def _configure_tesseract():
    """Set tesseract_cmd from config or well-known Windows paths."""
    try:
        import pytesseract
    except ImportError:
        return

    cmd = current_app.config.get("TESSERACT_CMD", "")
    if cmd and os.path.isfile(cmd):
        pytesseract.pytesseract.tesseract_cmd = cmd
        return

    # Auto-detect common Windows paths
    candidates = [
        r"D:\programme\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in candidates:
        if os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return


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

        # Fallback 1: pdfplumber — better at structured/tabular PDFs
        if not text:
            try:
                import pdfplumber
                with pdfplumber.open(str(path)) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
                text = text.strip()
            except Exception as exc:
                print(f"[pdfplumber] Extraction failed: {exc}")

        # Fallback 2: page-by-page OCR (scanned PDF)
        if not text:
            try:
                import fitz
                doc = fitz.open(str(path))
                for page_num in range(len(doc)):
                    pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(2, 2))
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
        """Dual-pass Tesseract OCR with OpenCV preprocessing."""
        try:
            import cv2
            import numpy as np
            import pytesseract

            _configure_tesseract()

            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Invalid image data")

            # Upscale + grayscale — 1.5x is enough for printed invoices without excessive RAM
            h, w = image.shape[:2]
            if max(h, w) < 2000:  # only upscale small images
                image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Set TESSDATA_PREFIX before doing anything else
            tessdata_dir = current_app.config.get("TESSDATA_DIR", "")
            if tessdata_dir and Path(tessdata_dir).exists():
                os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)
                pytesseract.pytesseract.environ["TESSDATA_PREFIX"] = str(tessdata_dir)

            requested = [l.strip() for l in current_app.config.get("TESSERACT_LANGUAGES", "eng").split("+") if l.strip()]

            # Safely check which langs are installed — never crash the whole request
            try:
                available = set(pytesseract.get_languages(config=f'--tessdata-dir "{tessdata_dir}"' if tessdata_dir else ""))
                languages = [l for l in requested if l in available]
            except Exception as lang_err:
                print(f"[Tesseract] lang check failed ({lang_err}), using requested: {requested}")
                languages = requested

            if not languages:
                languages = ["eng"]
            lang_str = "+".join(languages)
            print(f"[Tesseract] Using languages: {lang_str}")

            # Pass 1 — PSM 3: full automatic page segmentation (layout analysis)
            text_psm3 = pytesseract.image_to_string(gray, lang=lang_str, config="--oem 3 --psm 3")
            # Pass 2 — PSM 6: uniform block of text (line-by-line across columns)
            text_psm6 = pytesseract.image_to_string(gray, lang=lang_str, config="--oem 3 --psm 6")

            combined = (
                "=== OCR PASS 1 (Layout Analysis) ===\n"
                f"{text_psm3}\n\n"
                "=== OCR PASS 2 (Line-by-Line Analysis) ===\n"
                f"{text_psm6}"
            )
            return combined.strip()

        except ImportError as exc:
            raise RuntimeError("OpenCV and Tesseract are required for OCR") from exc
        except Exception as exc:
            print(f"[Tesseract] OCR failed: {exc}")
            return ""

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
                return current_app.config.get("OLLAMA_MODEL", "llama3.1")
            
            for preferred in ['llama3.1', 'llama3:latest', 'llama3', 'mistral', 'qwen', 'llama']:
                if preferred in models:
                    return preferred
                for m in models:
                    if preferred in m:
                        return m
            return models[0]
        except Exception as e:
            print(f"[Ollama] Failed to detect models: {e}")
            return current_app.config.get("OLLAMA_MODEL", "llama3.1")

    def extract_structured_data(self, text: str) -> tuple[dict[str, Any], str]:
        """Try Ollama first, fall back to spaCy NER, then regex heuristics."""

        # ── Ollama (primary) ──────────────────────────────────────────────────
        try:
            import ollama as ollama_lib

            system_prompt = (
                "Tu es un assistant d'extraction de données de factures extrêmement précis et rigoureux.\n"
                "Ton rôle est d'analyser le texte extrait (qui peut provenir d'un PDF ou de plusieurs passes d'OCR) pour remplir une structure de données.\n"
                "Consignes de rigueur absolue :\n"
                "1. Ne devine pas et ne calcule pas la TVA (vat_amount). Si la TVA/Taxes n'est pas écrite noir sur blanc dans le texte, sa valeur est null.\n"
                "2. Ne confonds jamais le fournisseur (supplier) et le client. Le fournisseur est l'émetteur du document (souvent en haut de page).\n"
                "3. Normalise document_type en une des valeurs suivantes uniquement : \"Facture\", \"Reçu\", \"Avoir\", \"Note\", \"Autre\".\n"
                "4. Nettoie les coquilles d'OCR évidentes (ex: \"US0\" en devises → \"USD\").\n"
                "5. Réponds uniquement sous la forme d'un objet JSON valide, sans Markdown."
            )

            user_prompt = (
                "Extrait les informations suivantes du texte d'OCR fourni :\n"
                "- supplier (string) : Nom du fournisseur / émetteur de la facture.\n"
                "- tax_identifier (string) : Identifiant fiscal (MF / RNE / TVA intra-UE) ou null.\n"
                "- invoice_number (string) : Numéro / référence du document.\n"
                "- invoice_date (string) : Date normalisée au format YYYY-MM-DD.\n"
                "- total_ht (string) : Montant hors taxes (Subtotal / Total HT).\n"
                "- vat_amount (string) : Montant de TVA / VAT. null si non mentionné explicitement.\n"
                "- stamp_amount (string) : Montant du timbre fiscal. null si absent.\n"
                "- total_ttc (string) : Montant toutes taxes comprises (TOTAL).\n"
                "- currency (string) : Code ISO devise (TND, EUR, USD…). Par défaut \"TND\".\n"
                "- document_type (string) : \"Facture\", \"Reçu\", \"Avoir\", \"Note\" ou \"Autre\".\n"
                "- category (string) : Catégorie métier stricte, choisis PARMI CETTE LISTE EXACTE :\n"
                "  [\"Achats de marchandises\", \"Matières premières\", \"Fournitures de bureau\", \"Informatique\", \"Télécommunication\", \"Eau, électricité, gaz\", \"Loyer\", \"Assurance\", \"Publicité et marketing\", \"Transport\", \"Déplacements\", \"Entretien et réparation\", \"Honoraires\", \"Formation\", \"Frais bancaires\", \"Impôts et taxes\", \"Salaires et charges sociales\", \"Immobilisations\"]\n"
                "  Si non applicable ou introuvable, retourne null.\n"
                "- confidence (object) : niveau de confiance par champ (low/medium/high).\n"
                "- missing_fields (array) : champs non trouvés.\n"
                "- warnings (array) : avertissements.\n\n"
                f"Texte extrait :\n\"\"\"\n{text[:6000]}\n\"\"\"\n\nJSON :"
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
            t.join(timeout=5)   # fail fast if Ollama is not running

            if _err_box:
                raise _err_box[0]
            if not _result_box:
                raise TimeoutError("Ollama did not respond within 30 seconds")

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
        mf_block = re.search(
            r"(.{0,80})\n[^\n]*(?:MF|matricule fiscale?|identifiant fiscal)[^\n]*",
            text, re.I
        )
        if mf_block:
            candidate = mf_block.group(1).strip().splitlines()[-1].strip()
            if 3 < len(candidate) < 80:
                return candidate
        for line in text.splitlines():
            line = line.strip()
            if 4 <= len(line) <= 70 and re.search(r"[A-Za-zÀ-ÿ]{3}", line) and not re.fullmatch(r"[\d\s.,:;/\\-]+", line):
                return line
        return None

    def _extract_tax_id(self, text: str) -> str | None:
        m = re.search(
            r"(?:MF|matricule\s+fiscale?|identifiant\s+fiscal|n[°o]?\s*TVA)[^\n:]*[:\s]+([A-Z0-9/\\]{5,25})",
            text, re.I
        )
        return m.group(1).strip() if m else None

    def _extract_invoice_number(self, text: str) -> str | None:
        patterns = [
            r"[Ff]acture\s+[Nn][°o]?\.?\s*([A-Z0-9][-A-Z0-9/]{3,})",
            r"[Ff]acture\s+[Nn][°o]?\.?\s*(\d{6,})",
            r"[Ii]nvoice\s+[Nn][uo]?\.?\s*([A-Z0-9][-A-Z0-9/]{3,})",
            r"[Rr][éeÉE]f[eé]rence\s*[:\s]+([A-Z0-9][A-Z0-9/_-]{3,})",
            r"N[°o]?\s+[Ff]acture\s*[:\s]+([A-Z0-9][-A-Z0-9/]{3,})",
            r"\bN[°o]\s*(\d{8,})\b",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(1).strip()
        return None

    def _extract_date(self, text: str) -> str | None:
        labelled = re.search(
            r"(?:date\s+(?:limite\s+de\s+)?(?:paiement|facture|[eé]mission|du?\s+document)[^\n]{0,30})"
            r"(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2})",
            text, re.I
        )
        target = labelled.group(1) if labelled else None
        if not target:
            m = re.search(r"\b(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2})\b", text)
            target = m.group(1) if m else None
        if not target:
            return None
        return self._normalise_date(target.replace(".", "/").replace("-", "/"))

    def _extract_labelled_amount(self, text: str, label_pattern: str) -> str | None:
        pattern = r"(?i)" + label_pattern + r"[^\n\d]{0,30}((?:\d{1,3}[,\s])*\d{1,3}[,.]\d{2,3})"
        m = re.search(pattern, text, re.I)
        return self._decimal_string(m.group(1)) if m else None

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
        if not value:
            return None
        normalized = (
            str(value)
            .replace("\u00a0", "")
            .replace("\u202f", "")
            .replace(" ", "")
            .replace("'", "")
        )
        if normalized.count(",") == 1 and normalized.count(".") == 0:
            normalized = normalized.replace(",", ".")
        elif "," in normalized and "." in normalized:
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
