import os
from flask import Flask, request, jsonify, send_file, send_from_directory, abort
from flask_cors import CORS
from flask_migrate import Migrate
from werkzeug.utils import secure_filename
from datetime import datetime
import pandas as pd
from io import BytesIO

from models import db, Invoice
from ocr_service import process_document

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.dirname(__file__), 'instance', 'invoices.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)

# Create tables on startup if they don't exist yet (for first run without migrations)
with app.app_context():
    os.makedirs(os.path.join(os.path.dirname(__file__), 'instance'), exist_ok=True)
    db.create_all()

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Upload & Process ──────────────────────────────────────────────────────────

@app.route('/api/upload', methods=['POST'])
def upload_file():
    # ── Basic validation ──────────────────────────────────────────────────────
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier partagé"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Aucun fichier sélectionné"}), 400

    if not (file and allowed_file(file.filename)):
        return jsonify({"error": "Type de fichier non autorisé. Formats acceptés : PDF, PNG, JPG, JPEG"}), 415

    # ── Save file ─────────────────────────────────────────────────────────────
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(filepath)
    except OSError as e:
        return jsonify({"error": f"Impossible de sauvegarder le fichier : {str(e)}"}), 500

    # ── OCR + AI Extraction ───────────────────────────────────────────────────
    try:
        extracted_data = process_document(filepath)
    except Exception as e:
        return jsonify({"error": f"Erreur lors du traitement du document : {str(e)}"}), 500

    if "error" in extracted_data:
        return jsonify(extracted_data), 500

    # ── Parse date safely ─────────────────────────────────────────────────────
    inv_date = None
    raw_date = extracted_data.get('invoice_date')
    if raw_date:
        for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%d.%m.%Y', '%d-%m-%Y'):
            try:
                inv_date = datetime.strptime(str(raw_date), fmt).date()
                break
            except ValueError:
                continue

    # ── Parse numeric fields safely ───────────────────────────────────────────
    def to_float(val):
        if val is None or val == '':
            return None
        if isinstance(val, str):
            val = val.replace('\u202f', '').replace('\xa0', '').replace(' ', '').replace(',', '.')
        try:
            result = float(val)
            return result if result >= 0 else None
        except (ValueError, TypeError):
            return None

    # ── Resolve currency (default TND) ───────────────────────────────────────
    raw_currency = extracted_data.get('currency')
    currency = raw_currency.strip().upper() if isinstance(raw_currency, str) and raw_currency.strip() else 'TND'

    # ── Save to database ──────────────────────────────────────────────────────
    try:
        new_invoice = Invoice(
            filename=filename,
            supplier=extracted_data.get('supplier') or 'Fournisseur Inconnu',
            invoice_number=extracted_data.get('invoice_number') or 'N/A',
            invoice_date=inv_date,
            total_ht=to_float(extracted_data.get('total_ht')),
            vat=to_float(extracted_data.get('vat')),
            total_ttc=to_float(extracted_data.get('total_ttc')),
            currency=currency,
            document_type=extracted_data.get('document_type') or 'Facture',
            extracted_text=extracted_data.get('extracted_text'),
            status='En attente'
        )
        db.session.add(new_invoice)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Erreur lors de l'enregistrement en base de données : {str(e)}"}), 500

    return jsonify({
        "message": "Fichier traité avec succès",
        "invoice": new_invoice.to_dict()
    }), 201


# ─── Serve uploaded files ───────────────────────────────────────────────────────

@app.route('/api/uploads/<path:filename>', methods=['GET'])
def serve_upload(filename):
    """Serve a previously uploaded document (image/PDF) for preview."""
    safe_name = secure_filename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    if not os.path.isfile(file_path):
        abort(404)
    return send_from_directory(app.config['UPLOAD_FOLDER'], safe_name)


# ─── CRUD ──────────────────────────────────────────────────────────────────────

@app.route('/api/invoices', methods=['GET'])
def get_invoices():
    invoices = db.session.execute(
        db.select(Invoice).order_by(Invoice.created_at.desc())
    ).scalars().all()
    return jsonify([inv.to_dict() for inv in invoices])



@app.route('/api/invoices/<int:invoice_id>', methods=['GET'])
def get_invoice(invoice_id):
    invoice = db.session.get(Invoice, invoice_id)
    if not invoice:
        return jsonify({"error": "Facture introuvable"}), 404
    return jsonify(invoice.to_dict())


@app.route('/api/invoices/<int:invoice_id>', methods=['PUT'])
def validate_invoice(invoice_id):
    invoice = db.session.get(Invoice, invoice_id)
    if not invoice:
        return jsonify({"error": "Facture introuvable"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "Corps de requête JSON manquant"}), 400

    def to_float(val):
        if val is None or val == '':
            return None
        if isinstance(val, str):
            val = val.replace('\u202f', '').replace('\xa0', '').replace(' ', '').replace(',', '.')
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # Update fields
    if 'supplier' in data:        invoice.supplier       = data['supplier'] or 'Fournisseur Inconnu'
    if 'invoice_number' in data:  invoice.invoice_number = data['invoice_number'] or 'N/A'
    if 'total_ht' in data:        invoice.total_ht       = to_float(data['total_ht'])
    if 'vat' in data:             invoice.vat            = to_float(data['vat'])
    if 'total_ttc' in data:       invoice.total_ttc      = to_float(data['total_ttc'])
    if 'currency' in data:
        raw = data['currency']
        invoice.currency = raw.strip().upper() if isinstance(raw, str) and raw.strip() else 'TND'
    if 'document_type' in data:   invoice.document_type  = data['document_type'] or 'Facture'
    if 'status' in data:          invoice.status         = data['status']

    if 'invoice_date' in data and data['invoice_date']:
        for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%d.%m.%Y'):
            try:
                invoice.invoice_date = datetime.strptime(data['invoice_date'], fmt).date()
                break
            except ValueError:
                continue

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Erreur lors de la mise à jour : {str(e)}"}), 500

    return jsonify(invoice.to_dict())


@app.route('/api/invoices/<int:invoice_id>', methods=['DELETE'])
def delete_invoice(invoice_id):
    invoice = db.session.get(Invoice, invoice_id)
    if not invoice:
        return jsonify({"error": "Facture introuvable"}), 404
    try:
        db.session.delete(invoice)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Erreur lors de la suppression : {str(e)}"}), 500
    return jsonify({"message": "Facture supprimée avec succès"}), 200


# ─── Export ────────────────────────────────────────────────────────────────────

@app.route('/api/export/excel', methods=['GET'])
def export_excel():
    invoices = db.session.execute(db.select(Invoice)).scalars().all()
    data = [inv.to_dict() for inv in invoices]

    df = pd.DataFrame(data)
    if not df.empty:
        df = df.drop(columns=['extracted_text', 'id'], errors='ignore')
        # Rename columns to French labels for the export
        df = df.rename(columns={
            'filename': 'Fichier',
            'supplier': 'Fournisseur',
            'invoice_number': 'N° Facture',
            'invoice_date': 'Date',
            'total_ht': 'Total HT',
            'vat': 'TVA',
            'total_ttc': 'Total TTC',
            'currency': 'Devise',
            'document_type': 'Type de document',
            'status': 'Statut',
            'created_at': 'Créé le'
        })

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Factures')

    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='factures_export.xlsx'
    )


@app.route('/api/export/csv', methods=['GET'])
def export_csv():
    invoices = db.session.execute(db.select(Invoice).order_by(Invoice.created_at.desc())).scalars().all()
    data = [inv.to_dict() for inv in invoices]

    df = pd.DataFrame(data)
    if not df.empty:
        df = df.drop(columns=['extracted_text', 'id'], errors='ignore')
        # Rename columns to French labels for the export
        df = df.rename(columns={
            'filename': 'Fichier',
            'supplier': 'Fournisseur',
            'invoice_number': 'N° Facture',
            'invoice_date': 'Date',
            'total_ht': 'Total HT',
            'vat': 'TVA',
            'total_ttc': 'Total TTC',
            'currency': 'Devise',
            'document_type': 'Type de document',
            'status': 'Statut',
            'created_at': 'Créé le'
        })

    output = BytesIO()
    # Using utf-8-sig so Excel recognizes the CSV encoding correctly
    csv_data = df.to_csv(index=False)
    output.write(csv_data.encode('utf-8-sig'))
    output.seek(0)

    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='factures_export.csv'
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
