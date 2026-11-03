# Implementation status

## Current milestone
Initial complete implementation is in place. The app supports CSV/XLSX uploads, preprocessing, sales analytics, recursive forecasts, model comparison, RFM clustering, user-entered inventory, and downloadable reports/templates.

## Files created
- `app.py`; modular `src/` data, modeling, inventory, segmentation, evaluation, and chart modules
- `tests/` preprocessing, forecasting, inventory, segmentation tests
- `docs/` project report, methodology, architecture
- `README.md`, `requirements.txt`, `.gitignore`, `TASKS.md`

## Data and modeling notes
- Dataset upload is user supplied; the app does not silently fetch UCI data.
- Inventory levels and lead times are collected separately and never inferred from transactions; reorder rates use the mean of a 30-day product forecast (or its labeled sparse-history fallback).
- Sales training aggregates positive non-cancelled transactions by product and day.
- For short series, a documented seasonal-naive/mean fallback is used; metrics are derived from chronological held-out data.
- RFM excludes transactions without a CustomerID; these remain in sales reporting.

## Verification
- Created a project-local `.venv` and installed `requirements.txt` (system user-site install was blocked by permissions).
- `.venv\Scripts\python.exe -m pytest -q`: **8 passed** (final run, 3.95 seconds).
- `.venv\Scripts\python.exe -m streamlit run app.py --server.headless true`: Streamlit started and reported local URL `http://localhost:8501`; stopped after confirming startup.
- Checks were performed without a supplied retail dataset; uploaded-data end-to-end metrics and forecasting depend on the user's dataset.

## Resume instructions
Install dependencies with `python -m pip install -r requirements.txt`, run tests, then start with `streamlit run app.py`. If the local environment lacks optional packages, install requirements and retry.
