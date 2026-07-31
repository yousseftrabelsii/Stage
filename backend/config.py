import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Application settings. Values may be overridden through a .env file."""

    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/invoice_assistant",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = str(BASE_DIR / "uploads")
    EXPORT_FOLDER = str(BASE_DIR / "exports")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    # Tesseract language codes, in priority order. Missing local language packs
    # are ignored so a French-only or Arabic-only workstation still works.
    TESSERACT_LANGUAGES = os.getenv("TESSERACT_LANGUAGES", "eng+fra+ara")
    TESSDATA_DIR = os.getenv("TESSDATA_DIR", str(BASE_DIR / "tessdata"))

    # Keep first local startup simple. Production should use `flask db upgrade`.
    AUTO_CREATE_DATABASE = os.getenv("AUTO_CREATE_DATABASE", "true").lower() == "true"
