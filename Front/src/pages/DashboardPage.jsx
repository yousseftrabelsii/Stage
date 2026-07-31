import React, { useState, useEffect } from 'react';
import { Container, Table, Button, Row, Col } from 'react-bootstrap';
import { FiDownload, FiSearch, FiFileText, FiEdit2, FiTrash2 } from 'react-icons/fi';
import { useNavigate } from 'react-router-dom';

const DashboardPage = () => {
  const [invoices, setInvoices] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    fetchInvoices();
  }, []);

  const fetchInvoices = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/invoices');
      const data = await response.json();
      setInvoices(data);
    } catch (e) {
      console.error("Error fetching invoices", e);
    }
  };

  const handleDelete = async (id) => {
    if (window.confirm("Voulez-vous vraiment supprimer cette facture ?")) {
      try {
        const response = await fetch(`http://localhost:5000/api/invoices/${id}`, {
          method: 'DELETE'
        });
        if (response.ok) {
          fetchInvoices();
        } else {
          alert("Erreur lors de la suppression");
        }
      } catch (e) {
        console.error(e);
      }
    }
  };

  const handleEdit = (inv) => {
    navigate('/validation', { state: { invoice: inv } });
  };

  return (
    <Container fluid className="px-4 py-4">
      <Row className="mb-4 align-items-end g-3">
        <Col md={6}>
          <h1 className="h3 fw-bold mb-1">Historique des factures</h1>
          <p className="text-muted mb-0">Visualisez et exportez les données extraites.</p>
        </Col>
        <Col md={6} className="d-flex justify-content-md-end gap-2">
          <Button 
            variant="outline-custom" 
            className="d-flex align-items-center gap-2"
            href="http://localhost:5000/api/export/csv"
          >
            <FiDownload /> Export CSV
          </Button>
          <Button 
            variant="outline-custom" 
            className="d-flex align-items-center gap-2"
            href="http://localhost:5000/api/export/excel"
          >
            <FiDownload /> Export Excel
          </Button>
        </Col>
      </Row>

      <div className="saas-card overflow-hidden">
        <div className="p-3 border-bottom d-flex justify-content-between align-items-center bg-white">
          <div className="position-relative" style={{ width: '320px' }}>
            <FiSearch className="position-absolute text-muted" style={{ left: '12px', top: '50%', transform: 'translateY(-50%)' }} />
            <input type="text" className="form-control ps-5" placeholder="Rechercher (fournisseur, n° facture)..." />
          </div>
        </div>
        
        <div className="table-responsive">
          <Table hover className="mb-0 align-middle">
            <thead>
              <tr>
                <th className="px-4">Fichier Source</th>
                <th>Fournisseur</th>
                <th>N° Facture</th>
                <th>Date</th>
                <th className="text-end">Montant TTC</th>
                <th className="text-center">Statut</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {invoices.map((inv) => (
                <tr key={inv.id}>
                  <td className="px-4 py-3">
                    <div className="d-flex align-items-center gap-2 text-dark">
                      <FiFileText className="text-muted" />
                      <span className="fw-medium text-truncate" style={{ maxWidth: '200px' }}>{inv.filename}</span>
                    </div>
                  </td>
                  <td className="py-3">{inv.supplier}</td>
                  <td className="py-3 text-muted">{inv.invoice_number}</td>
                  <td className="py-3 text-muted">{inv.invoice_date ? new Date(inv.invoice_date).toLocaleDateString('fr-FR') : 'N/A'}</td>
                  <td className="py-3 text-end fw-medium">
                    {inv.total_ttc ? inv.total_ttc.toFixed(2) : '0.00'} {inv.currency === 'EUR' ? '€' : (inv.currency === 'USD' ? '$' : (inv.currency === 'GBP' ? '£' : inv.currency))}
                  </td>
                  <td className="py-3 text-center">
                    <span className={`badge rounded-pill px-2 py-1 fw-medium ${inv.status === 'Validée' ? 'bg-success bg-opacity-10 text-success border border-success border-opacity-25' : 'bg-warning bg-opacity-10 text-warning border border-warning border-opacity-25'}`}>
                      {inv.status}
                    </span>
                  </td>
                  <td className="py-3 text-end pe-4">
                    <Button variant="link" className="text-primary p-1 me-2" onClick={() => handleEdit(inv)}>
                      <FiEdit2 />
                    </Button>
                    <Button variant="link" className="text-danger p-1" onClick={() => handleDelete(inv.id)}>
                      <FiTrash2 />
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </Table>
        </div>
      </div>
    </Container>
  );
};

export default DashboardPage;
