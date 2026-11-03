# Architecture

```text
Streamlit app.py
  ├── data_loader.py → schema normalization and CSV/XLSX ingestion
  ├── preprocessing.py → validity, cancellation, and return audit
  ├── feature_engineering.py → product/day series and lag features
  ├── sales_forecasting.py → chronological regressors, ensemble, recursive forecast
  ├── customer_segmentation.py → RFM and K-Means
  ├── inventory_management.py → user-input reorder calculations
  └── visualization.py → Plotly chart helper
```

Uploaded transactions are held in Streamlit session state. Cleaning is cached. Inventory data is provided separately by CSV or editor; there is no inventory inference from transactions. Model fitting occurs only on forecasting/evaluation actions, not on every dashboard render.
