import pandas as pd

def daily_product_sales(sales: pd.DataFrame) -> pd.DataFrame:
    if sales.empty:
        return pd.DataFrame(columns=["StockCode", "Date", "Sales"])
    x = sales.copy()
    x["Date"] = x["InvoiceDate"].dt.floor("D")
    products = []
    for product, group in x.groupby("StockCode", sort=False):
        days = pd.date_range(group["Date"].min(), group["Date"].max(), freq="D")
        series = group.groupby("Date")["Quantity"].sum().reindex(days, fill_value=0)
        products.append(pd.DataFrame({"StockCode": str(product), "Date": days, "Sales": series.to_numpy()}))
    return pd.concat(products, ignore_index=True) if products else pd.DataFrame(columns=["StockCode", "Date", "Sales"])

def make_features(series: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    values = series.astype(float)
    out = pd.DataFrame(index=range(len(values)))
    out["lag_1"] = values.shift(1)
    for lag in (7, 14, 28): out[f"lag_{lag}"] = values.shift(lag)
    out["rolling_7"] = values.shift(1).rolling(7, min_periods=1).mean()
    out["rolling_28"] = values.shift(1).rolling(28, min_periods=1).mean()
    out["day_of_week"] = dates.day_of_week
    out["month"] = dates.month
    return out
