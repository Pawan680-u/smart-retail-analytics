from io import BytesIO
from pathlib import Path
import pandas as pd

ALIASES = {
    "InvoiceNo": ["InvoiceNo", "Invoice", "invoice_no", "Invoice Number"],
    "StockCode": ["StockCode", "ProductID", "Product ID", "stock_code"],
    "Description": ["Description", "Product", "product_name"],
    "Quantity": ["Quantity", "Qty", "quantity"],
    "InvoiceDate": ["InvoiceDate", "Invoice Date", "Date", "invoice_date"],
    "UnitPrice": ["UnitPrice", "Unit Price", "Price", "unit_price"],
    "CustomerID": ["CustomerID", "Customer ID", "customer_id"],
    "Country": ["Country", "country"],
}
REQUIRED = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice"]

def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    lookup = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    for target, names in ALIASES.items():
        for name in names:
            if name.lower() in lookup:
                rename[lookup[name.lower()]] = target
                break
    df = df.rename(columns=rename)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    for col in ("Description", "CustomerID", "Country"):
        if col not in df:
            df[col] = pd.NA
    return df

def read_retail_file(file) -> pd.DataFrame:
    name = getattr(file, "name", str(file)).lower()
    if name.endswith(".csv"):
        raw = pd.read_csv(file)
    elif name.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(file)
    else:
        raise ValueError("Upload a .csv or .xlsx file")
    return normalize_columns(raw)

def load_uci_excel(path) -> pd.DataFrame:
    """Read an already downloaded UCI workbook; no implicit network access."""
    return normalize_columns(pd.read_excel(Path(path)))
