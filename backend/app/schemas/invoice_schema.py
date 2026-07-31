from marshmallow import Schema, fields


class InvoiceSchema(Schema):
    supplier = fields.String(required=False, allow_none=True)
    invoice_number = fields.String(required=False, allow_none=True)
    currency = fields.String(required=False, allow_none=True)
    total_ht = fields.Decimal(required=False, allow_none=True)
    tax_amount = fields.Decimal(required=False, allow_none=True)
    total_ttc = fields.Decimal(required=False, allow_none=True)
    category = fields.String(required=False, allow_none=True)
    status = fields.String(required=False, load_default="UPLOADED")
