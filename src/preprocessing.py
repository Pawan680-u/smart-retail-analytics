import pandas as pd
from .data_loader import normalize_columns

def preprocess_transactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = normalize_columns(frame)
    initial = len(df)
    df = df.drop_duplicates().copy()
    duplicate_count = initial - len(df)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", dayfirst=True)
    for c in ("Quantity", "UnitPrice"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["InvoiceNo"] = df["InvoiceNo"].astype("string").str.strip()
    df["StockCode"] = df["StockCode"].astype("string").str.strip()
    cancelled = df["InvoiceNo"].str.upper().str.startswith("C", na=False)
    returns = (df["Quantity"] < 0) & ~cancelled
    valid = df["InvoiceDate"].notna() & df["Quantity"].notna() & df["UnitPrice"].notna() & (df["UnitPrice"] > 0) & df["StockCode"].notna()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["IsCancelled"] = cancelled
    df["IsReturn"] = returns
    df["IsValid"] = valid
    sales = df[valid & ~cancelled & (df["Quantity"] > 0) & (df["Revenue"] > 0)].copy()
    return sales, {"input_rows": initial, "duplicates_removed": duplicate_count, "invalid_rows": int((~valid).sum()), "cancelled_rows": int(cancelled.sum()), "return_rows": int(returns.sum()), "sales_rows": len(sales), "returns": df[returns].copy(), "cancelled": df[cancelled].copy()}
