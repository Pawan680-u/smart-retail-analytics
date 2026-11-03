import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from .feature_engineering import make_features
from .model_evaluation import regression_metrics

MODEL_FACTORIES = {"Linear Regression": lambda: LinearRegression(), "Decision Tree": lambda: DecisionTreeRegressor(max_depth=8, min_samples_leaf=2, random_state=42), "Random Forest": lambda: RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=42, n_jobs=-1)}

def _feature_row(history, date):
    a = np.asarray(history, dtype=float)
    def lag(n): return a[-n] if len(a) >= n else (float(np.mean(a)) if len(a) else 0.0)
    return pd.DataFrame([{"lag_1": lag(1), "lag_7": lag(7), "lag_14": lag(14), "lag_28": lag(28), "rolling_7": float(np.mean(a[-7:])) if len(a) else 0.0, "rolling_28": float(np.mean(a[-28:])) if len(a) else 0.0, "day_of_week": date.day_of_week, "month": date.month}])

def forecast_product(daily: pd.DataFrame, product: str, horizon: int = 7):
    if horizon not in (7, 15, 30): raise ValueError("Forecast horizon must be 7, 15, or 30 days")
    part = daily[daily.StockCode.astype(str) == str(product)].sort_values("Date")
    if part.empty: raise ValueError("No sales history for the selected product")
    vals = part.Sales.astype(float).to_numpy()
    dates = pd.DatetimeIndex(part.Date)
    feats = make_features(pd.Series(vals), dates)
    dataset = feats.assign(target=vals).dropna()
    fallback = len(vals) < 40 or len(dataset) < 15
    metrics, fitted, importance = {}, {}, {}
    if not fallback:
        n = len(dataset)
        train_end = max(1, int(n * .70))
        val_end = max(train_end + 1, int(n * .85))
        val_end = min(val_end, n - 1)
        train, validation, test = dataset.iloc[:train_end], dataset.iloc[train_end:val_end], dataset.iloc[val_end:]
        validation_predictions, test_predictions = {}, {}
        for name, factory in MODEL_FACTORIES.items():
            model = factory().fit(train.drop(columns="target"), train.target)
            validation_predictions[name] = model.predict(validation.drop(columns="target"))
            test_predictions[name] = model.predict(test.drop(columns="target"))
            metrics[name] = regression_metrics(test.target, test_predictions[name])
            fitted[name] = factory().fit(dataset.drop(columns="target"), dataset.target)
            if name == "Random Forest": importance = dict(zip(dataset.drop(columns="target").columns, fitted[name].feature_importances_))
        errors = np.array([regression_metrics(validation.target, validation_predictions[n])["MAE"] for n in MODEL_FACTORIES])
        weights = 1 / np.maximum(errors, 1e-8); weights /= weights.sum()
        metrics["Ensemble"] = regression_metrics(test.target, np.average(np.column_stack([test_predictions[n] for n in MODEL_FACTORIES]), axis=1, weights=weights))
        # Validation-derived weights are used for future recursive prediction.
        weights_by_name = dict(zip(MODEL_FACTORIES, weights))
        model_name = "Ensemble"
    else:
        weights_by_name = {}
        model_name = "Seasonal-naive fallback" if len(vals) >= 7 else "Historical-mean fallback"
    hist = list(vals); future_rows = []
    for i in range(horizon):
        date = dates.max() + pd.Timedelta(days=i+1)
        row = _feature_row(hist, date)
        if fallback:
            pred = hist[-7] if len(hist) >= 7 else float(np.mean(hist))
        else:
            pred = sum(weights_by_name[n] * float(fitted[n].predict(row)[0]) for n in MODEL_FACTORIES)
        pred = max(0.0, pred); hist.append(pred)
        future_rows.append({"Date": date, "PredictedSales": pred})
    return {"history": part, "forecast": pd.DataFrame(future_rows), "metrics": metrics, "fallback": fallback, "model_name": model_name, "feature_importance": importance}

def compare_product_models(daily: pd.DataFrame, product: str):
    return forecast_product(daily, product, 7)["metrics"]
