# AI-Based Smart Retail Sales Prediction and Inventory Management System

## Abstract

This project provides a local web dashboard that converts historical online retail transactions into sales analytics, product-level daily demand forecasts, RFM-based customer groups, and explainable replenishment suggestions. It combines a Streamlit interface with Python data processing and scikit-learn models. Inventory levels and supplier terms are entered independently because transaction data does not contain those operational facts.

## Introduction and problem statement

Retail transaction histories are often available as spreadsheets but are difficult to translate into product demand and customer behavior decisions. Forecasting is complicated by cancellations, returns, sparse product histories, and time dependence. A practical system must clean these records, evaluate forecasts without random temporal leakage, and state the limits of inferred recommendations.

## Objectives

- Clean CSV and Excel retail transaction data.
- Summarize revenue, orders, units, products, and countries.
- Forecast product daily unit demand with regression and an ensemble.
- Segment identified customers with Recency, Frequency, and Monetary features.
- Compute reorder points from forecast/observed demand plus user-supplied stock details.
- Present results in a navigable dashboard with downloadable reports.

## Existing and proposed systems

Spreadsheet-only workflows require manual aggregation and do not provide repeatable model evaluation. The proposed local application centralizes the workflow, exposes data-quality counts, offers chronological model evaluation, and keeps operational inventory inputs explicit.

## Literature and background

The system applies established supervised regression, lag-based forecasting, K-Means clustering, and RFM customer analysis. The UCI Online Retail dataset is a widely used transaction dataset for exploratory retail analysis. The implementation is an academic decision-support system; no claim is made that one algorithm is universally superior.

## Dataset description

The UCI Online Retail dataset contains invoice, product, quantity, timestamp, unit price, customer, and country fields. It contains neither current stock nor lead times. Users can also supply compatible CSV/XLSX data. See `data/README.md` and the UCI source reference in the main README.

## Preprocessing and feature engineering

Schema variants are normalized, exact duplicates removed, dates/numbers parsed, cancellations and returns identified, and invalid rows excluded from positive-sales forecasts. Revenue equals quantity times unit price. Daily product unit demand is filled with zero only between observed product dates. Lag and rolling features use prior values; calendar features include weekday and month.

## Algorithms and evaluation

Linear Regression, Decision Tree, and Random Forest models are trained chronologically. A validation segment sets inverse-MAE ensemble weights; the final chronological segment measures MAE, RMSE, and R². Forecast recursion feeds predictions into future lag features. Short histories receive a transparent fallback. Customer groups use standardized log RFM features with deterministic K-Means; silhouette is shown where valid.

## System architecture and implementation

The Streamlit UI calls modular `src` components for ingestion, preprocessing, features, forecasting, segmentation, inventory, and charts. See `docs/ARCHITECTURE.md`. The app runs locally and requires no paid APIs or GPU.

## Results and discussion

Results depend on the uploaded file and are calculated at runtime. Dashboard metrics and evaluation scores must be interpreted against coverage and data quality. No benchmark values are asserted in this report; run the app on the selected dataset and capture its outputs for the final academic submission.

## Advantages

- Reproducible and inspectable data cleaning.
- Temporal rather than random evaluation.
- Multiple regression baselines and weighted ensemble.
- Explainable inventory formula and separate stock inputs.
- Interactive charts and downloadable CSV outputs.

## Limitations and future scope

Transaction history is observational and may omit promotions, stockouts, holidays, or catalog changes. Sparse products have weak forecasts; CustomerID gaps limit segmentation. Future work could add external holiday/promotion features, probabilistic prediction intervals, database-backed inventory snapshots, and monitoring after deployment.

## Conclusion

The project delivers an integrated academic prototype for retail analytics and decision support while making its data and inventory assumptions explicit. Real-world deployment would require validated operational data and ongoing monitoring.

## References

1. Chen, D. (2015). *Online Retail*. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
2. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*.
3. Scikit-learn documentation. https://scikit-learn.org/stable/
4. Streamlit documentation. https://docs.streamlit.io/
