# Backend — Assistant IA Factures

## Workflow API

1. `POST /api/uploads` : importer un PDF, JPG ou PNG.
2. `POST /api/documents/<id>/process` : extraction PDF/OCR, puis Ollama (`qwen2.5:3b`) vers JSON contrôlé.
3. `PUT /api/invoices/<id>` : corriger et valider les champs détectés.
4. `GET /api/exports?format=csv` ou `xlsx` : télécharger les factures validées.

Configurez `DATABASE_URL`, `OLLAMA_URL`, `OLLAMA_MODEL` et un `SECRET_KEY` robuste dans `.env`.
Utilisez PostgreSQL et les migrations (`flask db migrate`, puis `flask db upgrade`) hors développement local.
