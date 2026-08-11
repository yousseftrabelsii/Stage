import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(r"c:\Users\USER\Desktop\Stage\backend")

from app.services.extraction_service import ExtractionService

def test():
    service = ExtractionService()
    path = Path(r"c:\Users\USER\Desktop\Stage\backend\uploads\f7f9c504a884487388e0cdd9905a1581.pdf")
    
    print("Extracting text...")
    text = service.extract_text(path)
    print("--- OCR TEXT ---")
    print(text)
    print("----------------")
    
    print("Extracting structured data...")
    result, model = service.extract_structured_data(text)
    print(f"--- AI RESULT (Model: {model}) ---")
    import json
    print(json.dumps(result, indent=2))
    print("----------------")

if __name__ == "__main__":
    test()
