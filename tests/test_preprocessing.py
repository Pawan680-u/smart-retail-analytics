import pandas as pd
from src.preprocessing import preprocess_transactions

def sample():
    return pd.DataFrame({"InvoiceNo":["1","C2","3","4"],"StockCode":["A"]*4,"Quantity":[2,-1,-2,1],"InvoiceDate":["2024-01-01"]*4,"UnitPrice":[5,5,5,0]})

def test_filters_sales_and_tracks_returns_cancellations():
    sales, audit=preprocess_transactions(sample())
    assert len(sales)==1 and sales.iloc[0].Revenue==10
    assert audit["cancelled_rows"]==1 and audit["return_rows"]==1

def test_deduplicates_exact_rows():
    df=pd.concat([sample(),sample().iloc[[0]]],ignore_index=True)
    _, audit=preprocess_transactions(df)
    assert audit["duplicates_removed"]==1
