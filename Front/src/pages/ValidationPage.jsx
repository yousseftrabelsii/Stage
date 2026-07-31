import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Form, Button, Spinner, Alert } from 'react-bootstrap';
import { FiCheck, FiX, FiFileText, FiAlertCircle, FiImage } from 'react-icons/fi';
import { useNavigate, useLocation } from 'react-router-dom';

const API_BASE = 'http://localhost:5000';

const ValidationPage = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const [invoiceId, setInvoiceId] = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [success, setSuccess]     = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isPdf, setIsPdf]           = useState(false);

  const [formData, setFormData] = useState({
    supplier:       '',
    invoice_number: '',
    invoice_date:   '',
    total_ht:       '',
    vat:            '',
    total_ttc:      '',
    currency:       'EUR',
    document_type:  'Facture',
    filename:       '',
  });

  /* ── Load invoice from navigation state ─────────────────────────────────── */
  useEffect(() => {
    if (location.state && location.state.invoice) {
      const inv = location.state.invoice;
      setInvoiceId(inv.id);

      // Normalise the date: the DB sends "YYYY-MM-DD" which is what <input type="date"> expects.
      const rawDate = inv.invoice_date ?? '';

      setFormData({
        supplier:       inv.supplier       ?? '',
        invoice_number: inv.invoice_number ?? '',
        invoice_date:   rawDate,
        total_ht:       inv.total_ht  != null ? inv.total_ht  : '',
        vat:            inv.vat       != null ? inv.vat       : '',
        total_ttc:      inv.total_ttc != null ? inv.total_ttc : '',
        currency:       inv.currency      ?? 'EUR',
        document_type:  inv.document_type ?? 'Facture',
        filename:       inv.filename      ?? '',
      });

      // Build the preview URL
      if (inv.filename) {
        const url = `${API_BASE}/api/uploads/${encodeURIComponent(inv.filename)}`;
        setPreviewUrl(url);
        setIsPdf(inv.filename.toLowerCase().endsWith('.pdf'));
      }
    }
  }, [location.state]);

  /* ── Handlers ────────────────────────────────────────────────────────────── */
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleValidate = async () => {
    if (!invoiceId) {
      navigate('/dashboard');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const payload = {
        ...formData,
        status: 'Validée',
        // Send empty string as null for numeric fields
        total_ht:  formData.total_ht  !== '' ? parseFloat(formData.total_ht)  : null,
        vat:       formData.vat       !== '' ? parseFloat(formData.vat)       : null,
        total_ttc: formData.total_ttc !== '' ? parseFloat(formData.total_ttc) : null,
      };

      const response = await fetch(`${API_BASE}/api/invoices/${invoiceId}`, {
        method:  'PUT',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || `Erreur serveur (${response.status})`);
      }

      setSuccess('Facture validée avec succès !');
      setTimeout(() => navigate('/dashboard'), 1200);
    } catch (e) {
      setError(e.message || 'Erreur lors de la validation.');
    } finally {
      setLoading(false);
    }
  };

  const handleReject = async () => {
    if (!invoiceId) {
      navigate('/dashboard');
      return;
    }
    if (!window.confirm('Êtes-vous sûr de vouloir rejeter cette facture ? Elle sera supprimée.')) {
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/api/invoices/${invoiceId}`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || `Erreur serveur (${response.status})`);
      }
      navigate('/dashboard');
    } catch (e) {
      setError(e.message || 'Erreur lors du rejet.');
    } finally {
      setLoading(false);
    }
  };

  /* ── Render ──────────────────────────────────────────────────────────────── */
  return (
    <Container fluid className="px-4 py-4">
      {/* Header */}
      <div className="d-flex justify-content-between align-items-center mb-4 flex-wrap gap-3">
        <div>
          <h1 className="h3 fw-bold mb-1">Validation des données</h1>
          <p className="text-muted mb-0">
            Vérifiez les informations extraites par l&apos;OCR avant de les sauvegarder.
          </p>
        </div>
        <div className="d-flex gap-2">
          <Button
            id="btn-reject"
            variant="outline-danger"
            className="d-flex align-items-center gap-2"
            onClick={handleReject}
            disabled={loading || !invoiceId}
          >
            {loading ? <Spinner size="sm" animation="border" /> : <FiX />}
            Rejeter
          </Button>
          <Button
            id="btn-validate"
            variant="success"
            className="d-flex align-items-center gap-2"
            onClick={handleValidate}
            disabled={loading || !invoiceId}
          >
            {loading ? <Spinner size="sm" animation="border" /> : <FiCheck />}
            Valider et Enregistrer
          </Button>
        </div>
      </div>

      {/* Alerts */}
      {error && (
        <Alert variant="danger" dismissible onClose={() => setError(null)} className="d-flex align-items-center gap-2">
          <FiAlertCircle /> {error}
        </Alert>
      )}
      {success && (
        <Alert variant="success" className="d-flex align-items-center gap-2">
          <FiCheck /> {success}
        </Alert>
      )}
      {!invoiceId && (
        <Alert variant="warning" className="d-flex align-items-center gap-2">
          <FiAlertCircle /> Aucune facture sélectionnée.{' '}
          <span
            role="button"
            className="text-primary text-decoration-underline ms-1"
            onClick={() => navigate('/')}
          >
            Importer un document
          </span>
        </Alert>
      )}

      <Row className="g-4">
        {/* ── Left: Document preview ───────────────────────────── */}
        <Col lg={7}>
          <div className="saas-card h-100 d-flex flex-column">
            <div className="border-bottom px-4 py-3 bg-light rounded-top d-flex justify-content-between align-items-center">
              <span className="fw-semibold text-dark d-flex align-items-center gap-2">
                <FiFileText /> Document Source
              </span>
              <span className="text-muted small text-truncate ms-2" style={{ maxWidth: '260px' }}>
                {formData.filename || '—'}
              </span>
            </div>

            <div
              className="flex-grow-1 d-flex align-items-center justify-content-center bg-light"
              style={{ minHeight: '600px', overflow: 'hidden' }}
            >
              {previewUrl ? (
                isPdf ? (
                  /* PDF preview using <iframe> */
                  <iframe
                    title="Aperçu PDF"
                    src={previewUrl}
                    style={{ width: '100%', height: '600px', border: 'none' }}
                  />
                ) : (
                  /* Image preview */
                  <img
                    src={previewUrl}
                    alt="Aperçu du document"
                    style={{
                      maxWidth: '100%',
                      maxHeight: '600px',
                      objectFit: 'contain',
                      borderRadius: '4px',
                    }}
                    onError={() => setPreviewUrl(null)}
                  />
                )
              ) : (
                <div className="text-center text-muted p-4">
                  <FiImage size={52} className="mb-3 opacity-40" />
                  <p className="mb-0 fw-medium">Aperçu non disponible</p>
                  <p className="small">Le fichier n&apos;a pas pu être chargé.</p>
                </div>
              )}
            </div>
          </div>
        </Col>

        {/* ── Right: Extracted fields ──────────────────────────── */}
        <Col lg={5}>
          <div className="saas-card h-100">
            <div className="border-bottom px-4 py-3 bg-light rounded-top">
              <span className="fw-semibold text-dark">Données Extraites</span>
            </div>
            <div className="p-4">
              <Form>
                {/* Fournisseur */}
                <Form.Group className="mb-3">
                  <Form.Label className="small fw-semibold text-muted mb-1">Fournisseur</Form.Label>
                  <Form.Control
                    id="field-supplier"
                    type="text"
                    name="supplier"
                    value={formData.supplier}
                    onChange={handleChange}
                    placeholder="Nom du fournisseur"
                  />
                </Form.Group>

                {/* N° Facture + Date */}
                <Row className="g-3 mb-3">
                  <Col md={6}>
                    <Form.Label className="small fw-semibold text-muted mb-1">N° Facture</Form.Label>
                    <Form.Control
                      id="field-invoice-number"
                      type="text"
                      name="invoice_number"
                      value={formData.invoice_number}
                      onChange={handleChange}
                      placeholder="INV-XXXX"
                    />
                  </Col>
                  <Col md={6}>
                    <Form.Label className="small fw-semibold text-muted mb-1">Date</Form.Label>
                    <Form.Control
                      id="field-invoice-date"
                      type="date"
                      name="invoice_date"
                      value={formData.invoice_date}
                      onChange={handleChange}
                    />
                  </Col>
                </Row>

                {/* Type de document */}
                <Form.Group className="mb-4">
                  <Form.Label className="small fw-semibold text-muted mb-1">Type de document</Form.Label>
                  <Form.Select
                    id="field-document-type"
                    name="document_type"
                    value={formData.document_type}
                    onChange={handleChange}
                  >
                    <option value="Facture">Facture</option>
                    <option value="Reçu">Reçu</option>
                    <option value="Avoir">Avoir</option>
                    <option value="Note">Note</option>
                    <option value="Autre">Autre</option>
                  </Form.Select>
                </Form.Group>

                <hr className="my-4 text-muted" />

                {/* Montants */}
                <Row className="g-3">
                  <Col md={4}>
                    <Form.Label className="small fw-semibold text-muted mb-1">Montant HT</Form.Label>
                    <Form.Control
                      id="field-total-ht"
                      type="number"
                      step="0.01"
                      name="total_ht"
                      value={formData.total_ht}
                      onChange={handleChange}
                      className="text-end"
                      placeholder="0.00"
                    />
                  </Col>
                  <Col md={4}>
                    <Form.Label className="small fw-semibold text-muted mb-1">TVA</Form.Label>
                    <Form.Control
                      id="field-vat"
                      type="number"
                      step="0.01"
                      name="vat"
                      value={formData.vat}
                      onChange={handleChange}
                      className="text-end"
                      placeholder="0.00"
                    />
                  </Col>
                  <Col md={4}>
                    <Form.Label className="small fw-semibold text-muted mb-1">Montant TTC</Form.Label>
                    <Form.Control
                      id="field-total-ttc"
                      type="number"
                      step="0.01"
                      name="total_ttc"
                      value={formData.total_ttc}
                      onChange={handleChange}
                      className="text-end fw-semibold"
                      placeholder="0.00"
                    />
                  </Col>
                </Row>

                {/* Devise */}
                <Form.Group className="mt-3">
                  <Form.Label className="small fw-semibold text-muted mb-1">Devise</Form.Label>
                  <Form.Select
                    id="field-currency"
                    name="currency"
                    value={formData.currency}
                    onChange={handleChange}
                  >
                    <option value="EUR">EUR (€)</option>
                    <option value="USD">USD ($)</option>
                    <option value="TND">TND</option>
                    <option value="GBP">GBP (£)</option>
                    <option value="MAD">MAD</option>
                    <option value="CAD">CAD</option>
                  </Form.Select>
                </Form.Group>
              </Form>
            </div>
          </div>
        </Col>
      </Row>
    </Container>
  );
};

export default ValidationPage;
