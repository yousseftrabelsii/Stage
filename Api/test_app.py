import io
import pytest
from app import app, db
from models import Invoice


@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'

    with app.test_client() as client:
        with app.app_context():
            db.drop_all()
            db.create_all()
        yield client
        with app.app_context():
            db.session.remove()
            db.drop_all()


# ─── GET /api/invoices ────────────────────────────────────────────────────────

def test_get_invoices_empty(client):
    response = client.get('/api/invoices')
    assert response.status_code == 200
    assert response.get_json() == []


# ─── POST /api/upload ─────────────────────────────────────────────────────────

def test_upload_missing_file(client):
    response = client.post('/api/upload')
    assert response.status_code == 400
    assert "error" in response.get_json()

def test_upload_empty_filename(client):
    data = {'file': (io.BytesIO(b""), "")}
    response = client.post('/api/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert "error" in response.get_json()

def test_upload_invalid_file_type(client):
    data = {'file': (io.BytesIO(b"dummy content"), "test.txt")}
    response = client.post('/api/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert response.get_json()["error"] == "Type de fichier non autorisé"


# ─── PUT /api/invoices/<id> ───────────────────────────────────────────────────

def test_update_nonexistent_invoice(client):
    response = client.put('/api/invoices/9999',
                          json={"status": "Validée"},
                          content_type='application/json')
    assert response.status_code == 404

def test_update_invoice(client):
    with app.app_context():
        inv = Invoice(filename='test.pdf', supplier='Acme', status='En attente')
        db.session.add(inv)
        db.session.commit()
        inv_id = inv.id

    response = client.put(f'/api/invoices/{inv_id}',
                          json={"supplier": "Acme Corp", "status": "Validée"},
                          content_type='application/json')
    assert response.status_code == 200
    data = response.get_json()
    assert data['supplier'] == 'Acme Corp'
    assert data['status'] == 'Validée'


# ─── DELETE /api/invoices/<id> ────────────────────────────────────────────────

def test_delete_nonexistent_invoice(client):
    response = client.delete('/api/invoices/9999')
    assert response.status_code == 404

def test_delete_invoice(client):
    with app.app_context():
        inv = Invoice(filename='delete_me.pdf', status='En attente')
        db.session.add(inv)
        db.session.commit()
        inv_id = inv.id

    response = client.delete(f'/api/invoices/{inv_id}')
    assert response.status_code == 200

    # Verify it's gone
    response = client.get('/api/invoices')
    assert all(i['id'] != inv_id for i in response.get_json())


# ─── GET /api/export/excel ────────────────────────────────────────────────────

def test_export_excel_empty(client):
    response = client.get('/api/export/excel')
    assert response.status_code == 200
    assert 'spreadsheetml' in response.content_type
