import React from 'react';
import { NavLink, Link } from 'react-router-dom';
import { Container, Nav, Navbar as BootstrapNavbar } from 'react-bootstrap';
import { FiBox, FiUpload, FiCheckSquare, FiList } from 'react-icons/fi';

const Navbar = () => {
  return (
    <BootstrapNavbar expand="lg" className="bg-white border-bottom sticky-top py-2">
      <Container fluid className="px-4">
        <BootstrapNavbar.Brand as={Link} to="/" className="d-flex align-items-center gap-2">
          <FiBox size={24} className="text-primary" />
          <span>SmartInvoice</span>
        </BootstrapNavbar.Brand>
        <BootstrapNavbar.Toggle aria-controls="basic-navbar-nav" />
        <BootstrapNavbar.Collapse id="basic-navbar-nav">
          <Nav className="ms-4 gap-2">
            <Nav.Link as={NavLink} to="/" className="d-flex align-items-center gap-2 rounded px-3">
              <FiUpload size={16} /> Import
            </Nav.Link>
            <Nav.Link as={NavLink} to="/validation" className="d-flex align-items-center gap-2 rounded px-3">
              <FiCheckSquare size={16} /> Validation
            </Nav.Link>
            <Nav.Link as={NavLink} to="/dashboard" className="d-flex align-items-center gap-2 rounded px-3">
              <FiList size={16} /> Historique
            </Nav.Link>
          </Nav>
        </BootstrapNavbar.Collapse>
      </Container>
    </BootstrapNavbar>
  );
};

export default Navbar;
