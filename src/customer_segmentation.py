import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def build_rfm(sales: pd.DataFrame, n_clusters: int = 4):
    x = sales.dropna(subset=["CustomerID"]).copy()
    if x.empty: return pd.DataFrame(), {"silhouette": None, "message": "CustomerID is unavailable; RFM segmentation cannot be computed."}
    x["CustomerID"] = x.CustomerID.astype(str)
    anchor = x.InvoiceDate.max().normalize() + pd.Timedelta(days=1)
    rfm = x.groupby("CustomerID").agg(Recency=("InvoiceDate", lambda d: (anchor-d.max().normalize()).days), Frequency=("InvoiceNo", "nunique"), Monetary=("Revenue", "sum")).reset_index()
    rfm[["Recency", "Frequency", "Monetary"]] = rfm[["Recency", "Frequency", "Monetary"]].clip(lower=0)
    if len(rfm) < 3:
        rfm["Cluster"] = 0; rfm["Segment"] = "Insufficient customers for clustering"
        return rfm, {"silhouette": None, "message": "At least 3 identified customers are needed for K-Means."}
    k = min(max(2, int(n_clusters)), min(6, len(rfm)-1))
    values = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    scaled = StandardScaler().fit_transform(values)
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    rfm["Cluster"] = labels
    profiles = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    med_frequency, med_money = rfm.Frequency.median(), rfm.Monetary.median()
    names = {}
    for cluster, p in profiles.iterrows():
        if p.Monetary >= med_money and p.Frequency >= med_frequency and p.Recency <= rfm.Recency.median(): name = "High Value Customers"
        elif p.Recency > rfm.Recency.median() and p.Frequency >= med_frequency: name = "At-Risk Customers"
        elif p.Frequency >= med_frequency: name = "Regular Customers"
        else: name = "Potential Customers"
        names[cluster] = name
    rfm["Segment"] = rfm.Cluster.map(names)
    score = silhouette_score(scaled, labels) if 1 < len(set(labels)) < len(rfm) else None
    return rfm, {"silhouette": float(score) if score is not None else None, "profiles": profiles, "message": f"K-Means fitted with {k} clusters; names are assigned from observed RFM profiles."}
