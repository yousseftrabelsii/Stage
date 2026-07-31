import os
import cv2
import pytesseract
import fitz  # PyMuPDF
import pdfplumber
import spacy
import json
import ollama
import re

# ─── Tesseract path (Windows) ──────────────────────────────────────────────────
# Auto-detect Tesseract installation
_TESSERACT_PATHS = [
    r'D:\programme\Tesseract-OCR\tesseract.exe',
]
for _path in _TESSERACT_PATHS:
    if os.path.isfile(_path):
        pytesseract.pytesseract.tesseract_cmd = _path
        break
else:
    print("[WARNING] Tesseract OCR not found. Install it from: https://github.com/UB-Mannheim/tesseract/wiki")

# ─── spaCy model ───────────────────────────────────────────────────────────────
# Uses French NER model; falls back to a blank model if not yet downloaded.
# To install: python -m spacy download fr_core_news_sm
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    nlp = spacy.blank("fr")


# ─── Text extraction ──────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF using PyMuPDF, with pdfplumber as fallback."""
    text = ""

    # Primary: PyMuPDF (fast, works well on text-based PDFs)
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
        text = text.strip()
    except Exception as e:
        print(f"[PyMuPDF] Extraction failed: {e}")
        text = ""

    # Fallback: pdfplumber (better at structured/tabular PDFs)
    if not text:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            text = text.strip()
        except Exception as e:
            print(f"[pdfplumber] Extraction failed: {e}")

    return text


def extract_text_from_image(image_path: str) -> str:
    """Extract text from an image using OpenCV preprocessing + two-pass Tesseract OCR."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Preprocessing: upscale → grayscale (omit Otsu thresholding as it breaks slashes/decimals)
        image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Pass 1: PSM 3 (Layout analysis)
        custom_config_psm3 = r'--oem 3 --psm 3'
        text_psm3 = pytesseract.image_to_string(gray, lang='fra+eng', config=custom_config_psm3)

        # Pass 2: PSM 6 (Line-by-line analysis across columns)
        custom_config_psm6 = r'--oem 3 --psm 6'
        text_psm6 = pytesseract.image_to_string(gray, lang='fra+eng', config=custom_config_psm6)

        combined_text = (
            "=== OCR PASS 1 (Layout Analysis) ===\n"
            f"{text_psm3}\n\n"
            "=== OCR PASS 2 (Line-by-Line Analysis) ===\n"
            f"{text_psm6}"
        )
        return combined_text.strip()
    except Exception as e:
        print(f"[Tesseract] OCR failed: {e}")
        return ""


# ─── Structured extraction ────────────────────────────────────────────────────

def fallback_spacy_extraction(text: str) -> dict:
    """
    Fallback extractor using spaCy NER.
    Used when Ollama is unavailable or returns invalid JSON.
    """
    doc = nlp(text)

    supplier = "Fournisseur Inconnu"
    for ent in doc.ents:
        if ent.label_ in ("ORG", "PER"):
            supplier = ent.text
            break

    # Regex: find all decimal amounts (e.g. 1 234,56 or 1234.56)
    amounts = re.findall(r'\b\d[\d\s]*[\.,]\d{2}\b', text)
    def parse_amount(s):
        s = s.replace(' ', '').replace(',', '.')
        try:
            return float(s)
        except ValueError:
            return 0.0

    parsed = [parse_amount(a) for a in amounts]
    total_ttc = max(parsed) if parsed else 0.0

    # Regex: find dates in common formats
    date_match = re.search(
        r'\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}\.\d{2}\.\d{4})\b', text
    )
    invoice_date = None
    if date_match:
        raw = date_match.group(1)
        for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%d.%m.%Y'):
            try:
                from datetime import datetime
                invoice_date = datetime.strptime(raw, fmt).strftime('%Y-%m-%d')
                break
            except ValueError:
                continue

    return {
        "supplier":       supplier,
        "invoice_number": "N/A",
        "invoice_date":   invoice_date,
        "total_ht":       0.0,
        "vat":            0.0,
        "total_ttc":      total_ttc,
        "currency":       "TND",
        "document_type":  "Facture",
    }


def _get_best_ollama_model() -> str:
    """Detect available Ollama models and pick the best one."""
    try:
        models_info = ollama.list()
        models = []
        if hasattr(models_info, 'models'):
            models = [m.model for m in models_info.models]
        elif isinstance(models_info, dict) and 'models' in models_info:
            models = [m.get('model') if isinstance(m, dict) else getattr(m, 'model', None) for m in models_info['models']]
            models = [m for m in models if m]
        
        if not models:
            return 'llama3.1'  # default fallback if list empty
        
        # Preferred order
        for preferred in ['llama3.1', 'llama3:latest', 'llama3', 'mistral']:
            if preferred in models:
                return preferred
            for m in models:
                if preferred in m:
                    return m
        return models[0]
    except Exception as e:
        print(f"[Ollama] Error detecting models: {e}")
        return 'llama3.1'


def analyze_text_with_ollama(extracted_text: str) -> dict:
    """
    Send the extracted text to a local Ollama LLM and parse the JSON response.
    Falls back to spaCy-based extraction if Ollama is unavailable or fails.
    """
    system_prompt = (
        "Tu es un assistant d'extraction de données de factures extrêmement précis et rigoureux.\n"
        "Ton rôle est d'analyser le texte extrait (qui peut provenir d'un PDF ou de plusieurs passes d'OCR) pour remplir une structure de données.\n"
        "Consignes de rigueur absolue :\n"
        "1. Ne devine pas et ne calcule pas la TVA (vat). Si la TVA/Taxes n'est pas écrite noir sur blanc dans le texte, sa valeur est 0.0.\n"
        "2. Ne confonds jamais le fournisseur (supplier) et le client (buyer/importer). Le fournisseur est l'émetteur du document (souvent en haut de page).\n"
        "3. Normalise le type de document (document_type) en une des valeurs suivantes uniquement : \"Facture\", \"Reçu\", \"Avoir\", \"Note\", \"Autre\".\n"
        "4. Nettoie les coquilles d'OCR évidentes (ex: \"uso\" ou \"US0\" en devises devient \"USD\").\n"
        "5. Réponds uniquement sous la forme d'un objet JSON valide."
    )

    user_prompt = f"""Extrait les informations suivantes du texte d'OCR fourni ci-dessous :
- supplier (string) : Nom du fournisseur/émetteur de la facture (ex: GLOBAL TRADE EXPORT LTD.).
- invoice_number (string) : Numéro/référence du document.
- invoice_date (string) : Date normalisée au format YYYY-MM-DD.
- total_ht (float) : Montant hors taxes (Subtotal / Total HT).
- vat (float) : Montant de TVA / VAT. Doit être 0.0 s'il n'est pas mentionné explicitement sous le nom 'TVA' ou 'VAT' ou 'Tax' dans le texte.
- total_ttc (float) : Montant toutes taxes comprises (TOTAL).
- currency (string) : Code ISO de la devise (EUR, USD, TND, etc.).
- document_type (string) : "Facture", "Reçu", "Avoir", "Note" ou "Autre".

Texte extrait :
\"\"\"
{extracted_text[:4000]}
\"\"\"

JSON :"""

    try:
        model_name = _get_best_ollama_model()
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            format='json',
            options={'temperature': 0.0}  # Temp 0 for maximum consistency and precision
        )
        content = response.message.content.strip()

        # Extract JSON using regex in case of surrounding text
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group(0)

        return json.loads(content)

    except Exception as e:
        print(f"[Ollama] Failed, using spaCy fallback: {e}")
        return fallback_spacy_extraction(extracted_text)


def _normalize_currency(data: dict) -> dict:
    """
    Ensure the currency field is a clean ISO code.
    Defaults to 'TND' if the value is absent, None, empty, or not a string.
    """
    raw = data.get('currency')
    if isinstance(raw, str) and raw.strip():
        data['currency'] = raw.strip().upper()
    else:
        data['currency'] = 'TND'
    return data


# ─── Main entry point ─────────────────────────────────────────────────────────

def process_document(filepath: str) -> dict:
    """
    Full pipeline:
      1. Extract raw text (PyMuPDF → pdfplumber for PDFs, OpenCV+Tesseract for images)
      2. Analyze text with Ollama (spaCy fallback if Ollama is down)
      3. Return structured invoice data dict
    """
    ext = os.path.splitext(filepath)[1].lower()
    extracted_text = ""

    if ext == '.pdf':
        extracted_text = extract_text_from_pdf(filepath)
        # If the PDF appears to be a scanned image, try OCR page-by-page
        if not extracted_text:
            try:
                doc = fitz.open(filepath)
                for page_num in range(len(doc)):
                    pix = doc[page_num].get_pixmap(dpi=200)
                    img_path = filepath + f"_page{page_num}.png"
                    pix.save(img_path)
                    extracted_text += extract_text_from_image(img_path) + "\n"
                    os.remove(img_path)
                doc.close()
                extracted_text = extracted_text.strip()
            except Exception as e:
                print(f"[Scanned PDF OCR] Failed: {e}")

    elif ext in ('.png', '.jpg', '.jpeg'):
        extracted_text = extract_text_from_image(filepath)

    if not extracted_text:
        print(f"[process_document] No text extracted from {filepath}. Using placeholder.")
        extracted_text = "[Texte non lisible — OCR non configuré ou document vide]"

    structured_data = analyze_text_with_ollama(extracted_text)
    structured_data = _normalize_currency(structured_data)   # ← default TND if not detected
    structured_data['extracted_text'] = extracted_text
    return structured_data
