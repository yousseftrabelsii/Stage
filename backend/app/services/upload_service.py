from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from flask import current_app
from werkzeug.utils import secure_filename

from app.repositories.invoice_repository import InvoiceRepository


class UploadService:
    def __init__(self, repository: InvoiceRepository | None = None):
        self.repository = repository or InvoiceRepository()

    def save_uploaded_file(self, file_storage):
        original_name = secure_filename(file_storage.filename or "")
        if not original_name:
            raise ValueError("A valid filename is required")

        extension = Path(original_name).suffix.lower().lstrip(".")
        if extension not in current_app.config["ALLOWED_EXTENSIONS"]:
            raise ValueError("Unsupported file type. Use PDF, PNG, JPG or JPEG.")

        payload = file_storage.read()
        if not payload:
            raise ValueError("The uploaded file is empty")
        file_storage.stream.seek(0)
        safe_name = f"{uuid.uuid4().hex}.{extension}"
        upload_folder = Path(current_app.config["UPLOAD_FOLDER"])
        upload_folder.mkdir(parents=True, exist_ok=True)
        storage_path = upload_folder / safe_name
        storage_path.write_bytes(payload)

        document = self.repository.create_document(
            filename=safe_name,
            original_name=original_name,
            storage_path=str(storage_path),
            content_type=file_storage.mimetype or f"application/{extension}",
            file_size=len(payload),
            file_hash=hashlib.sha256(payload).hexdigest(),
        )
        invoice = self.repository.create_invoice(document)
        return document, invoice
