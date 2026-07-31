import os
from flask import Flask

from config import Config
from extensions import db, migrate

from app.models.invoice import Document, ExtractionRun, Invoice, InvoiceFieldValue
from app.routes.upload import upload_bp
from app.routes.invoice import invoice_bp
from app.routes.export import export_bp


def create_app(config_object=Config):
    app = Flask(__name__)

    app.config.from_object(config_object)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["EXPORT_FOLDER"], exist_ok=True)
    os.makedirs("instance", exist_ok=True)

    db.init_app(app)
    migrate.init_app(app, db)

    app.register_blueprint(upload_bp)
    app.register_blueprint(invoice_bp)
    app.register_blueprint(export_bp)

    # Flask-Migrate owns schema changes in shared environments. This fallback is
    # limited to local bootstrap and the test configuration.
    if app.config.get("AUTO_CREATE_DATABASE", False):
        with app.app_context():
            db.create_all()

    @app.route("/")
    def home():
        return {
            "message": "Assistant IA Backend is running",
            "version": "1.0",
        }

    @app.route("/health")
    def health():
        return {"status": "ok"}

    return app
