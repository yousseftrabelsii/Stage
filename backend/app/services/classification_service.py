from __future__ import annotations


class ClassificationService:
    def classify(self, text: str):
        return {"classification": "invoice", "text": text}
