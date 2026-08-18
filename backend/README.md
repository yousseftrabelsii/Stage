# Backend — Assistant IA Factures

Ce backend gère l'extraction intelligente des factures (via PaddleOCR, Tesseract, et Ollama) et expose une API REST pour le frontend React.

## 🚀 Installation & Démarrage (Guide Complet)

Si vous installez ce projet pour la première fois (notamment sur Windows), suivez **attentivement** ces étapes :

### 1. Prérequis Système
- **Python 3.10+** installé.
- **Tesseract OCR** : 
  - Téléchargez et installez Tesseract pour Windows (ex: via le site de l'Université de Mannheim).
  - Assurez-vous d'installer les langues supplémentaires : **Français** (`fra`), **Anglais** (`eng`), et **Arabe** (`ara`).
  - Notez le chemin d'installation (généralement `C:\Program Files\Tesseract-OCR\tesseract.exe`).
- **Ollama** :
  - Téléchargez Ollama depuis [ollama.com](https://ollama.com).
  - Lancez Ollama et téléchargez le modèle requis dans votre terminal : `ollama run qwen2.5:3b` (ou le modèle spécifié dans votre `.env`).

### 2. Variables d'Environnement (`.env`)
Créez un fichier `.env` à la racine du dossier `backend` et copiez-y la configuration suivante. **N'oubliez pas d'adapter le chemin Tesseract !**
```ini
SECRET_KEY=change-me-in-production
DATABASE_URL=sqlite:///instance/invoice_assistant.db # Ou votre URL PostgreSQL
UPLOAD_FOLDER=uploads
EXPORT_FOLDER=exports

# Modèle Ollama (Llama3 ou Qwen2.5)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b

# Configuration Tesseract (CRITIQUE SUR WINDOWS)
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
TESSERACT_LANGUAGES=fra+eng
```

### 3. Installation des Dépendances Python
Ouvrez un terminal dans le dossier `backend` et exécutez :

```bash
# 1. Créer un environnement virtuel (recommandé)
python -m venv venv
venv\Scripts\activate

# 2. Installer les paquets (attention aux versions PaddlePaddle/PaddleOCR)
pip install -r requirements.txt

# Si l'installation de PaddlePaddle échoue sur Windows, forcez cette version :
pip install paddlepaddle==3.3.1 paddleocr==3.7.0

# 3. Télécharger le modèle linguistique de base (spaCy) pour le fallback
python -m spacy download fr_core_news_md
```
*(Note: PaddleOCR nécessite les redistribuables Microsoft C++ de Visual Studio).*

### 4. Lancement du Serveur
Une fois tout installé, lancez le serveur Flask :
```bash
python run.py
```
Le backend sera disponible sur `http://localhost:5000`.

---

## ⚙️ Workflow API

1. `POST /api/uploads` : Import d'un document PDF, JPG ou PNG.
2. L'API lance un prétraitement d'image OpenCV intensif (Denoising, CLAHE, Deskew, Suppression de grilles).
3. **PaddleOCR** (ou **Tesseract** en fallback) extrait le texte.
4. L'API contacte **Ollama** avec un système de regex "Pre-Hints" pour extraire les données dans un schéma JSON strict.
5. `PUT /api/invoices/<id>` : Validation des champs par l'utilisateur.
6. `GET /api/exports?format=csv` ou `xlsx` : Export final des factures.
