import pandas as pd
from src.customer_segmentation import build_rfm

def test_missing_customer_ids_reported():
    df=pd.DataFrame({"CustomerID":[None],"InvoiceDate":pd.to_datetime(["2024-01-01"]),"InvoiceNo":["1"],"Revenue":[5.]})
    rfm, info=build_rfm(df)
    assert rfm.empty and "CustomerID" in info["message"]

def test_small_identified_cohort_has_clear_message():
    df=pd.DataFrame({"CustomerID":["1","2"],"InvoiceDate":pd.to_datetime(["2024-01-01","2024-01-02"]),"InvoiceNo":["1","2"],"Revenue":[5.,8.]})
    rfm, info=build_rfm(df)
    assert len(rfm)==2 and info["silhouette"] is None
