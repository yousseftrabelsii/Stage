import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Application settings. Values may be overridden through a .env file."""

    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")

    # ── Database ──────────────────────────────────────────────────────────────
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/invoice_assistant",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ── File storage ──────────────────────────────────────────────────────────
    UPLOAD_FOLDER = str(BASE_DIR / os.getenv("UPLOAD_FOLDER", "uploads"))
    EXPORT_FOLDER = str(BASE_DIR / os.getenv("EXPORT_FOLDER", "exports"))
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}

    # ── Ollama LLM ────────────────────────────────────────────────────────────
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

    # ── Tesseract OCR ─────────────────────────────────────────────────────────
    # Language codes for Tesseract (priority order; missing packs are ignored).
    TESSERACT_LANGUAGES = os.getenv("TESSERACT_LANGUAGES", "fra+ara+eng")
    TESSDATA_DIR = os.getenv("TESSDATA_DIR", str(BASE_DIR / "tessdata"))
    # Full path to the tesseract binary (required on Windows).
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")

    # ── Misc ──────────────────────────────────────────────────────────────────
    # Keep first local startup simple. Production should use `flask db upgrade`.
    AUTO_CREATE_DATABASE = os.getenv("AUTO_CREATE_DATABASE", "true").lower() == "true"
