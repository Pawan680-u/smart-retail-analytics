# Methodology

## Data preparation

Column aliases are normalized to the expected transaction schema. Exact duplicate rows are removed; dates and numeric values are parsed; rows with invalid dates, missing product codes, nonnumeric quantities/prices, or nonpositive prices are invalid. Invoice identifiers beginning with `C` are cancellation records. Negative quantities are returns. Both are retained in an audit structure and excluded from positive demand training. Positive sales revenue is `Quantity × UnitPrice`.

## Forecast features and validation

Demand is aggregated to product and calendar day, with missing days between a product's first and last transaction set to zero. Features include 1, 7, 14, and 28 day lags, 7/28 day lagged rolling averages, weekday, and month. The first 70% of feature rows trains models, the next 15% sets ensemble weights, and the last chronological 15% is the test set. No future target is used to construct a feature. The Random Forest importance is derived from the fitted model. Future predictions are recursive and clipped at zero. Products with fewer than 40 days of observations use a clearly reported seasonal-naive or historical-mean fallback.

## Algorithms

Linear Regression, a bounded-depth Decision Tree Regressor, and Random Forest Regressor are fit per selected product. An inverse validation-MAE weighted ensemble combines them. MAE, RMSE, and R² are computed on the final chronological test segment. Metrics are not hardcoded.

## RFM and inventory

For identified customers, Recency is days since last purchase relative to one day after the latest transaction, Frequency is distinct invoices, and Monetary is positive sales revenue. Log transformed and standardized RFM values feed deterministic K-Means. Cluster names are assigned from observed cluster profiles. The silhouette score is reported only when mathematically valid.

Inventory demand rate is estimated from the selected product's next 30 forecast days (including the explicitly labeled fallback when history is sparse). Reorder Point = mean forecast daily units × user-entered lead time + user-entered safety stock. Suggested quantity is max(0, reorder point − current stock), raised to the user minimum order quantity only when an order is required. For lead times beyond the forecast window, the 30-day forecast average is extrapolated. These are decision-support estimates, not claims about actual stock or future supply.
