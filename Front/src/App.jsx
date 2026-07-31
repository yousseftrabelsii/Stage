import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import UploadPage from './pages/UploadPage';
import ValidationPage from './pages/ValidationPage';
import DashboardPage from './pages/DashboardPage';
import { useEffect } from 'react';

function App() {
  // Set theme based on system preference initially
  useEffect(() => {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      document.documentElement.setAttribute('data-bs-theme', 'dark');
    } else {
      document.documentElement.setAttribute('data-bs-theme', 'light');
    }
  }, []);

  return (
    <div className="app-container">
      <Navbar />
      <main className="container py-5 animate-fade-in">
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/validation" element={<ValidationPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
