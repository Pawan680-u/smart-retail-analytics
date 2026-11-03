import pandas as pd

INVENTORY_COLUMNS = ["StockCode", "CurrentStock", "LeadTimeDays", "SafetyStock", "MinimumOrderQty"]

def inventory_template():
    return pd.DataFrame(columns=INVENTORY_COLUMNS)

def recommend_inventory(inventory: pd.DataFrame, daily_sales: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    required = set(INVENTORY_COLUMNS)
    if not required.issubset(inventory.columns): raise ValueError("Inventory requires columns: " + ", ".join(INVENTORY_COLUMNS))
    if horizon < 1: raise ValueError("horizon must be positive")
    inv = inventory.copy()
    for c in INVENTORY_COLUMNS[1:]: inv[c] = pd.to_numeric(inv[c], errors="coerce")
    if inv[INVENTORY_COLUMNS[1:]].isna().any().any() or (inv[INVENTORY_COLUMNS[1:]] < 0).any().any(): raise ValueError("Inventory quantities, lead times, and safety stock must be non-negative numbers")
    means = daily_sales.groupby(daily_sales.StockCode.astype(str)).Sales.mean().to_dict() if not daily_sales.empty else {}
    output=[]
    for row in inv.itertuples(index=False):
        pid, stock, lead, safety, minimum = row
        rate = float(means.get(str(pid), 0.0)); demand = rate * float(lead); reorder = demand + float(safety)
        needed = max(0.0, reorder-float(stock))
        suggested = max(needed, float(minimum)) if needed > 0 else 0.0
        if stock <= safety: status="Critical Stock"
        elif stock <= reorder: status="Reorder Required"
        elif stock <= reorder + max(safety, rate*7): status="Low Stock"
        else: status="In Stock"
        output.append({"StockCode":str(pid), "CurrentStock":stock, "AverageDailyDemand":rate, "LeadTimeDemand":demand, "SafetyStock":safety, "ReorderPoint":reorder, "SuggestedOrderQty":suggested, "Status":status, "Explanation":f"{rate:.2f} units/day × {lead:g} lead-time days + {safety:g} safety stock; minimum order {minimum:g}."})
    return pd.DataFrame(output)
