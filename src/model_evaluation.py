import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {"MAE": float(mean_absolute_error(actual, predicted)), "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))), "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan")}
