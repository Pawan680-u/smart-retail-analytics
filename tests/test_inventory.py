import pandas as pd
import pytest
from src.inventory_management import recommend_inventory

def test_reorder_point_and_minimum_order():
    inv=pd.DataFrame([{"StockCode":"A","CurrentStock":2,"LeadTimeDays":5,"SafetyStock":3,"MinimumOrderQty":10}])
    daily=pd.DataFrame({"StockCode":["A","A"],"Sales":[2,4]})
    result=recommend_inventory(inv,daily).iloc[0]
    assert result.ReorderPoint==18
    assert result.SuggestedOrderQty==16
    assert result.Status=="Critical Stock"

def test_negative_stock_input_rejected():
    inv=pd.DataFrame([{"StockCode":"A","CurrentStock":-1,"LeadTimeDays":2,"SafetyStock":1,"MinimumOrderQty":0}])
    with pytest.raises(ValueError): recommend_inventory(inv,pd.DataFrame())
