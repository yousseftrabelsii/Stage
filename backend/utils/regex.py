import re


def extract_invoice_number(text: str) -> str:
    match = re.search(r"(?:facture|invoice)[^\d]*(\d+)", text, re.IGNORECASE)
    return match.group(1) if match else ""
