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
    """Return a cached PaddleOCR instance (loads models only once).
    Compatible with PaddleOCR 3.x API.
    """
    global _paddle_ocr
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR
            # PaddleOCR 3.x API:
            _paddle_ocr = PaddleOCR(
                use_textline_orientation=True,
                lang="fr",
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

        # Primary: pdfplumber — preserves visual layout and table structures better
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    # extract_text(layout=True) preserves spacing and visual alignment
                    extracted = page.extract_text(layout=True)
                    if extracted:
                        text += extracted + "\n"
            text = text.strip()
        except Exception as exc:
            print(f"[pdfplumber] Extraction failed: {exc}")
            text = ""

        # Fallback 1: PyMuPDF — fast, but messes up tables
        if not text:
            try:
                import fitz
                doc = fitz.open(str(path))
                for page in doc:
                    text += page.get_text("text")
                doc.close()
                text = text.strip()
            except Exception as exc:
                print(f"[PyMuPDF] Extraction failed: {exc}")
                text = ""

        # Fallback 2: page-by-page OCR (scanned PDF)
        if not text or len(text) < 50:
            text = ""
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

            # ── 1. Document Cropping (Smart Scan) ──
            image = self._smart_crop(image)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # ── Denoise ──
            gray = cv2.fastNlMeansDenoising(gray, h=10)

            # ── CLAHE — improve local contrast for faded/uneven scans ──
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # ── 2. Grid & Line Removal ──
            # Remove horizontal/vertical lines that confuse OCR
            gray = self._remove_lines_and_noise(gray)

            # ── Sharpen — improves blurry/photocopied documents ──
            kernel_sharpen = np.array([[-1, -1, -1],
                                        [-1,  9, -1],
                                        [-1, -1, -1]])
            gray = cv2.filter2D(gray, -1, kernel_sharpen)

            # ── Deskew ── (straighten rotated scans up to ±10°)
            gray = self._deskew(gray)

            # Reconstruct a color (BGR) enhanced image from deskewed gray for PaddleOCR
            # PaddleOCR performs best on color/grayscale images (not aggressively binarized)
            enhanced_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # ── 3. Adaptive Binarization & 4. Morphological Erosion ──
            # Tesseract performs better on binarized images. We use Adaptive Thresholding
            # and a light erosion (which thickens black text) for faint characters.
            binary_for_tess = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
            )
            kernel_morph = np.ones((1, 1), np.uint8)
            binary_for_tess = cv2.erode(binary_for_tess, kernel_morph, iterations=1)

            # ── Run PaddleOCR ──
            ocr = _get_paddle_ocr()
            if ocr is not None:
                try:
                    # PaddleOCR 3.x: use predict() — ocr() with cls kwarg is removed
                    result = list(ocr.predict(enhanced_bgr))

                    # ── Flatten results into text lines ──
                    # v3 returns an iterable of OCRResult objects with .rec_texts attribute
                    lines: list[str] = []
                    for page in result:
                        if page is None:
                            continue
                        # v3 OCRResult object
                        if hasattr(page, 'rec_texts'):
                            for text_line in page.rec_texts:
                                if text_line and text_line.strip():
                                    lines.append(text_line.strip())
                        else:
                            # Legacy v2 fallback: list of [box_coords, (text, score)]
                            for box in page:
                                try:
                                    text_info = box[1]
                                    if isinstance(text_info, (list, tuple)):
                                        text_line = text_info[0]
                                    else:
                                        text_line = str(text_info)
                                    text_line = text_line.strip()
                                    if text_line:
                                        lines.append(text_line)
                                except (IndexError, TypeError):
                                    continue

                    recognized_text = "\n".join(lines).strip()
                    if recognized_text:
                        print(f"[PaddleOCR] Extracted {len(lines)} lines.")
                        return recognized_text
                    print("[PaddleOCR] No text found, trying Tesseract fallback.")
                except Exception as paddle_exc:
                    print(f"[PaddleOCR] predict() failed: {paddle_exc} — falling back to Tesseract.")
            else:
                print("[PaddleOCR] Not available, falling back to Tesseract.")

            # ── Tesseract fallback ──
            try:
                import pytesseract
                from flask import current_app
                from PIL import Image
                import io

                cmd = current_app.config.get("TESSERACT_CMD", "")
                if cmd:
                    pytesseract.pytesseract.tesseract_cmd = cmd

                lang = current_app.config.get("TESSERACT_LANGUAGES", "fra+eng")

                # Feed Tesseract the binarized and morphologically thickened image
                pil_img = Image.fromarray(binary_for_tess)
                tess_text = pytesseract.image_to_string(
                    pil_img,
                    lang=lang,
                    config="--oem 3 --psm 6",
                ).strip()
                print(f"[Tesseract] Extracted {len(tess_text.splitlines())} lines.")
                return tess_text
            except Exception as tess_exc:
                print(f"[Tesseract] Fallback also failed: {tess_exc}")
                return ""

        except ImportError as exc:
            raise RuntimeError("OpenCV is required for OCR") from exc
        except Exception as exc:
            import traceback
            print(f"[OCR] Failed: {exc}")
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

    @staticmethod
    def _smart_crop(image):
        """Automatically crop the image to the document contours, like a mobile scanner app."""
        try:
            import cv2
            import numpy as np
            
            # Resize for faster edge detection
            h, w = image.shape[:2]
            ratio = h / 500.0
            resized = cv2.resize(image, (int(w / ratio), 500))
            
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(gray, 75, 200)
            
            cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
            
            screenCnt = None
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    screenCnt = approx
                    break
                    
            if screenCnt is None:
                return image
                
            # Ensure contour is large enough (e.g. > 10% of image area)
            if cv2.contourArea(screenCnt) < (resized.shape[0] * resized.shape[1] * 0.1):
                return image
                
            # Map back to original image size
            pts = screenCnt.reshape(4, 2) * ratio
            
            # Order points (top-left, top-right, bottom-right, bottom-left)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")
                
            M = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        except Exception as e:
            print(f"[Smart Crop] Failed, using original: {e}")
            return image

    @staticmethod
    def _remove_lines_and_noise(gray):
        """Remove horizontal/vertical lines that confuse OCR."""
        try:
            import cv2
            import numpy as np
            
            # Binarize the image (invert so text/lines are white)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 21, 10)
            
            # Detect horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            
            # Detect vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # Combine lines
            lines = cv2.add(detect_horizontal, detect_vertical)
            
            # Thicken lines a bit for better masking
            kernel = np.ones((3, 3), np.uint8)
            lines = cv2.dilate(lines, kernel, iterations=1)
            
            # Inpaint to remove the lines from the original grayscale image
            cleaned = cv2.inpaint(gray, lines, 3, cv2.INPAINT_TELEA)
            return cleaned
        except Exception as e:
            print(f"[Line Removal] Failed, using original: {e}")
            return gray

    # ── Structured data extraction ─────────────────────────────────────────────

    def _guess_category_from_text(self, text: str) -> str | None:
        """Keyword-based category detection using Tunisian SCE (Plan Comptable) codes."""
        text_lower = text.lower()
        # Keys must exactly match the allowed_categories whitelist and frontend options.
        keywords: dict[str, list[str]] = {
            # 601 / 607 — Stock pour revente, matières premières, composants
            "Marchandises & Matières": [
                "marchandise", "revente", "grossiste", "stock", "matière première",
                "composant", "article", "bois", "acier", "farine", "ciment", "fer",
            ],
            # 6021 / 6022 — Électricité (STEG), Eau (SONEDE)
            "Énergie & Fluides": [
                "électricité", "steg", "eau", "sonede", "gaz", "énergie", "fluide",
                "carburant", "gasoil", "essence", "agil", "total énergies",
            ],
            # 605 — Outillage, habillement professionnel, matériel léger
            "Petits Équipements": [
                "outillage", "outil", "habillement professionnel", "équipement léger",
                "matériel de bureau", "petit matériel", "clé", "perceuse",
            ],
            # 6022 — Papier, stylos, cartouches d'encre (Consommables)
            "Fournitures de Bureau": [
                "fourniture", "papier", "stylo", "cartouche", "encre", "papeterie",
                "imprimante", "rame", "classeur", "cahier", "consommable",
            ],
            # 613 / 612 — Loyer de l'entrepôt, leasing automobile ou matériel
            "Loyers & Leasing": [
                "loyer", "location", "bail", "entrepôt", "leasing", "crédit-bail",
                "agence immobilière", "magasin",
            ],
            # 615 — Maintenance informatique, réparation de véhicule
            "Entretien & Réparations": [
                "entretien", "réparation", "maintenance", "dépannage",
                "pièce de rechange", "garage", "vidange", "révision",
            ],
            # 616 — Primes d'assurances (RC Pro, auto, locaux)
            "Assurances": [
                "assurance", "prime d'assurance", "responsabilité civile",
                "sinistre", "mutuelle", "star assurances", "gat", "comar", "ami assurance",
            ],
            # 622 / 617 — Factures d'expert-comptable, avocat, études
            "Honoraires & Conseil": [
                "honoraire", "expert-comptable", "comptable", "avocat", "notaire",
                "consultant", "conseil", "audit", "étude", "expertise", "retenue à la source",
            ],
            # 623 — Campagnes Web, impression de flyers, cadeaux
            "Publicité & Marketing": [
                "publicité", "marketing", "facebook ads", "google ads", "affiche",
                "campagne", "promotion", "sponsor", "flyer", "cadeau",
            ],
            # 625 — Billets d'avion (Tunisair), hôtels, missions
            "Voyages & Déplacements": [
                "déplacement", "billet d'avion", "tunisair", "hôtel", "mission",
                "voyage", "hébergement", "train", "taxi", "transport",
            ],
            # 625 — Factures de restaurants avec des partenaires
            "Repas & Réceptions": [
                "restaurant", "repas", "réception", "dîner", "déjeuner", "buffet",
                "traiteur", "café",
            ],
            # 626 — Internet, forfaits mobiles (TT, Ooredoo, Orange), timbres
            "Télécoms & Courrier": [
                "télécom", "internet", "forfait", "mobile", "tunisie telecom",
                "ooredoo", "orange", "fibre", "adsl", "timbre", "courrier", "poste",
            ],
            # 65 — Intérêts de crédits, agios, commissions de compte
            "Frais Bancaires": [
                "frais bancaire", "commission bancaire", "agios", "intérêt",
                "tenue de compte", "biat", "amen bank", "atb", "stb", "bh",
            ],
            # Classe 2 — Ordinateurs, voitures de fonction, meubles (> 500 DT HT)
            "Immobilisations": [
                "immobilisation", "ordinateur", "voiture de fonction", "meuble",
                "machine", "mobilier", "bâtiment", "matériel informatique",
                "investissement", "logiciel",
            ],
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
                "R7. NUMÉRO DE DOCUMENT (Facture/Devis) : correspond au 'invoice_number' dans le JSON. Inclure les préfixes/suffixes (ex: 'devis - 2026001', 'FAC-2024-001'). Ne pas tronquer.\n"
                "R8. MATRICULE FISCAL tunisien : format typique '1234567A/B/C/000' ou '1234567X/A/P/000'. Retourne-le tel quel.\n"
                "R9. document_type : exactement l'une de ces valeurs : 'Facture', 'Devis', 'Reçu', 'Avoir', 'Note', 'Autre'.\n"
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
                '  "invoice_number": "Numéro complet de la facture ou du devis ou null",\n'
                '  "invoice_date": "YYYY-MM-DD ou null",\n'
                '  "total_ht": "montant en string ex: 1500.000 ou null",\n'
                '  "vat_amount": "montant TVA en string ou null (jamais calculé)",\n'
                '  "stamp_amount": "montant timbre en string ou null",\n'
                '  "total_ttc": "montant TTC en string ou null",\n'
                '  "currency": "TND par défaut, sinon EUR/USD/etc.",\n'
                '  "document_type": "Facture|Devis|Reçu|Avoir|Note|Autre",\n'
                '  "category": "catégorie parmi la liste ou null",\n'
                '  "confidence": {"supplier":"high","invoice_number":"medium",...},\n'
                '  "missing_fields": ["champs absents"],\n'
                '  "warnings": ["anomalies détectées"]\n'
                "}\n\n"
                "Catégories autorisées UNIQUEMENT : "
                "Marchandises & Matières, Énergie & Fluides, Petits Équipements, Fournitures de Bureau, "
                "Loyers & Leasing, Entretien & Réparations, Assurances, Honoraires & Conseil, "
                "Publicité & Marketing, Voyages & Déplacements, Repas & Réceptions, "
                "Télécoms & Courrier, Frais Bancaires, Immobilisations.\n\n"
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
            t.join(timeout=180)   # 180s — llama3/qwen on CPU can be slow

            if _err_box:
                raise _err_box[0]
            if not _result_box:
                raise TimeoutError("Ollama did not respond within 180 seconds")

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
            r"[Dd]evis\s*[-:\s]+\s*([A-Z0-9][-A-Z0-9/]{2,})",
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

        if clean.get("invoice_date"):
            clean["invoice_date"] = self._normalise_date(clean["invoice_date"])

        raw_currency = clean.get("currency")
        clean["currency"] = str(raw_currency or "TND").strip().upper()[:10] or "TND"

        # Normalise document_type
        allowed_types = {"Facture", "Devis", "Reçu", "Avoir", "Note", "Autre"}
        dt = clean.get("document_type")
        clean["document_type"] = dt if dt in allowed_types else "Facture"
        
        # Normalise category — must match SCE (Plan Comptable Tunisien) labels
        allowed_categories = {
            "Marchandises & Matières",   # 601/607
            "Énergie & Fluides",          # 6021/6022
            "Petits Équipements",         # 605
            "Fournitures de Bureau",       # 6022
            "Loyers & Leasing",           # 613/612
            "Entretien & Réparations",    # 615
            "Assurances",                 # 616
            "Honoraires & Conseil",        # 622/617
            "Publicité & Marketing",       # 623
            "Voyages & Déplacements",      # 625
            "Repas & Réceptions",          # 625
            "Télécoms & Courrier",         # 626
            "Frais Bancaires",             # 65
            "Immobilisations",             # Classe 2
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
        value = str(value).strip()
        if not value:
            return None

        import re
        from datetime import date

        # 1. Try exact ISO first (YYYY-MM-DD)
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError:
            pass

        # 2. Try DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
        d_match = re.match(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$", value)
        if d_match:
            try:
                return date(int(d_match.group(3)), int(d_match.group(2)), int(d_match.group(1))).isoformat()
            except ValueError:
                pass

        # 3. Try YYYY/MM/DD, YYYY-MM-DD, YYYY.MM.DD
        y_match = re.match(r"^(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})$", value)
        if y_match:
            try:
                return date(int(y_match.group(1)), int(y_match.group(2)), int(y_match.group(3))).isoformat()
            except ValueError:
                pass

        # 4. Try DD/MM/YY
        yy_match = re.match(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{2})$", value)
        if yy_match:
            try:
                year = 2000 + int(yy_match.group(3))
                return date(year, int(yy_match.group(2)), int(yy_match.group(1))).isoformat()
            except ValueError:
                pass
                
        # 5. Try dateutil parser as a last resort
        try:
            from dateutil import parser
            dt = parser.parse(value, dayfirst=True)
            return dt.date().isoformat()
        except Exception:
            return None
