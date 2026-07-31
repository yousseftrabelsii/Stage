"""Test extraction parsing edge cases."""
from decimal import Decimal
from app.services.extraction_service import ExtractionService

svc = ExtractionService()

cases = [
    "76,000", "76.000", "76 000,000", "1.234,56", "1,234.56",
    "91,816", "15.816", "1 234,567", "1234", "0,600",
]
print("=== Decimal parsing ===")
for c in cases:
    print(f"  {c!r:15} -> {svc._decimal_string(c)!r}")

print("\n=== Date parsing ===")
dates = ["06/10/2022", "2022-10-06", "10/06/2022", "06-10-22", "31/12/2023"]
for d in dates:
    print(f"  {d!r:15} -> {svc._normalise_date(d)!r}")

print("\n=== Heuristic on Tunisian-like text ===")
text = """
STEG
Matricule fiscal: 1234567A/M/000
Facture N° 76241339
Date: 06/10/2022
Total HT: 76,000 DT
TVA 19%: 15,816
Timbre fiscal: 0,600
Total TTC: 91,816 TND
"""
result = svc._heuristic_extract(text)
for k, v in result.items():
    if k not in ("confidence", "missing_fields", "warnings"):
        print(f"  {k}: {v}")
