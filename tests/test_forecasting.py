import pandas as pd
import pytest
from src.sales_forecasting import forecast_product

def test_short_history_uses_labeled_fallback():
    d=pd.DataFrame({"StockCode":["P"]*5,"Date":pd.date_range("2024-01-01",periods=5),"Sales":[1,2,3,4,5]})
    result=forecast_product(d,"P",7)
    assert result["fallback"] and len(result["forecast"])==7
    assert (result["forecast"].PredictedSales>=0).all()

def test_invalid_horizon_rejected():
    d=pd.DataFrame({"StockCode":["P"],"Date":pd.to_datetime(["2024-01-01"]),"Sales":[1]})
    with pytest.raises(ValueError): forecast_product(d,"P",9)
