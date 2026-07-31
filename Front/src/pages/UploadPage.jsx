import React, { useState } from 'react';
import { Container, Row, Col, ProgressBar } from 'react-bootstrap';
import { FiUpload, FiFileText, FiCheck, FiSettings, FiDatabase } from 'react-icons/fi';
import { useNavigate } from 'react-router-dom';

const UploadPage = () => {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState(null);
  const [processingState, setProcessingState] = useState(0); 
  const navigate = useNavigate();

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = async (selectedFile) => {
    setFile(selectedFile);
    await uploadToBackend(selectedFile);
  };

  const uploadToBackend = async (selectedFile) => {
    setProcessingState(1); // Upload
    
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Step 2 & 3: Extraction and Preprocessing (Simulated UI step, actual happens on backend)
      setTimeout(() => setProcessingState(2), 500);
      setTimeout(() => setProcessingState(3), 1500);

      const response = await fetch('http://localhost:5000/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to upload");
      }

      const data = await response.json();
      
      setProcessingState(4); // Classification
      setTimeout(() => setProcessingState(5), 500); // Saving

      setTimeout(() => {
        setProcessingState(6);
        // Pass the extracted invoice data to the validation page
        navigate('/validation', { state: { invoice: data.invoice } });
      }, 1000);

    } catch (error) {
      console.error("Error uploading file:", error);
      alert("Erreur lors de l'envoi au backend. Vérifiez que le serveur Flask tourne.");
      setProcessingState(0);
    }
  };

  const steps = [
    { id: 1, name: "Téléchargement", icon: <FiUpload /> },
    { id: 2, name: "Prétraitement", icon: <FiSettings /> },
    { id: 3, name: "Extraction OCR", icon: <FiFileText /> },
    { id: 4, name: "Classification", icon: <FiDatabase /> },
    { id: 5, name: "Enregistrement", icon: <FiCheck /> }
  ];

  return (
    <Container className="py-5" style={{ maxWidth: '800px' }}>
      <div className="mb-5">
        <h1 className="h3 fw-bold mb-2">Traitement des factures</h1>
        <p className="text-muted">Importez une facture (PDF ou image) pour démarrer le processus d'extraction.</p>
      </div>

      <div className="saas-card p-4">
        {processingState === 0 ? (
          <div 
            className={`drop-zone d-flex flex-column align-items-center justify-content-center ${dragActive ? "active" : ""}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <div className="bg-light rounded-circle p-3 mb-3">
              <FiUpload size={24} className="text-secondary" />
            </div>
            <h5 className="fw-semibold mb-1">Sélectionnez un fichier ou glissez-le ici</h5>
            <p className="text-muted small mb-4">Formats acceptés : PDF, JPG, PNG (Max 10MB)</p>
            
            <input
              type="file"
              id="file-upload"
              className="d-none"
              accept=".pdf,.png,.jpg,.jpeg"
              onChange={handleChange}
            />
            <label htmlFor="file-upload" className="btn btn-primary-custom">
              Parcourir les fichiers
            </label>
          </div>
        ) : (
          <div className="p-2">
            <div className="d-flex justify-content-between align-items-center mb-3">
              <div>
                <h5 className="fw-semibold mb-1">Traitement en cours...</h5>
                <p className="text-muted small mb-0">{file?.name}</p>
              </div>
              <span className="badge bg-light text-dark border">
                {Math.round((processingState / 5) * 100)}%
              </span>
            </div>
            
            <ProgressBar 
              now={(processingState / 5) * 100} 
              className="mb-4" 
              style={{ height: '6px' }}
            />

            <div className="d-flex flex-column gap-2 mt-4">
              {steps.map(step => (
                <div key={step.id} className="d-flex align-items-center p-2">
                  <div className={`rounded-circle d-flex align-items-center justify-content-center me-3 ${processingState >= step.id ? 'bg-primary text-white' : 'bg-light text-muted'}`} style={{ width: '32px', height: '32px' }}>
                    {processingState > step.id ? <FiCheck size={14} /> : step.icon}
                  </div>
                  <div className={`flex-grow-1 ${processingState >= step.id ? 'fw-medium text-dark' : 'text-muted'}`}>
                    {step.name}
                  </div>
                  {processingState === step.id && (
                    <div className="spinner-border spinner-border-sm text-primary" role="status">
                      <span className="visually-hidden">Loading...</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Container>
  );
};

export default UploadPage;
