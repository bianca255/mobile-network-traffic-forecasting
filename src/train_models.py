from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100.0
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def prepare_feature_target(df: pd.DataFrame, target_col: str = "traffic") -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c not in {"timestamp", "traffic", "is_event"}]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


def fit_and_score(model, X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, y_eval: pd.Series) -> Dict[str, float]:
    model.fit(X_train, y_train)
    pred = model.predict(X_eval)
    return compute_metrics(np.asarray(y_eval), np.asarray(pred))


def run_experiments(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> List[dict]:
    """Run iterative experiments with a baseline plus tuned tree models."""
    records: List[dict] = []

    # Baseline: linear regression
    linear_model = LinearRegression()
    linear_metrics = fit_and_score(linear_model, X_train, y_train, X_test, y_test)
    records.append(
        {
            "stage": "baseline",
            "model": "LinearRegression",
            "params": {"fit_intercept": True},
            "metrics": linear_metrics,
            "rationale": "Interpretable baseline to quantify the simplest predictive signal from lagged temporal features.",
        }
    )

    # Experiment 1: RandomForest with moderate depth, tuned using validation RMSE
    rf_candidates = [
        {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 2, "random_state": 42},
        {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 1, "random_state": 42},
        {"n_estimators": 600, "max_depth": 16, "min_samples_leaf": 2, "random_state": 42},
    ]
    rf_best = None
    rf_best_val_rmse = float("inf")
    for params in rf_candidates:
        model = RandomForestRegressor(**params)
        val_pred = model.fit(X_train, y_train).predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(np.asarray(y_val), np.asarray(val_pred)))
        if val_rmse < rf_best_val_rmse:
            rf_best = params
            rf_best_val_rmse = val_rmse

    best_rf = RandomForestRegressor(**rf_best)
    rf_metrics = fit_and_score(best_rf, X_train, y_train, X_test, y_test)
    records.append(
        {
            "stage": "tuned",
            "model": "RandomForestRegressor",
            "params": rf_best,
            "metrics": rf_metrics,
            "validation_rmse": float(rf_best_val_rmse),
            "rationale": "The random forest captures nonlinear dependencies and threshold effects that are common in traffic surges, while validation RMSE guides the depth and leaf-size trade-off.",
        }
    )

    # Experiment 2: XGBoost tuned using validation RMSE
    xgb_candidates = [
        {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.8, "random_state": 42},
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42},
        {"n_estimators": 800, "max_depth": 7, "learning_rate": 0.02, "subsample": 0.9, "colsample_bytree": 0.9, "random_state": 42},
    ]
    xgb_best = None
    xgb_best_val_rmse = float("inf")
    for params in xgb_candidates:
        model = XGBRegressor(**params, objective="reg:squarederror")
        val_pred = model.fit(X_train, y_train).predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(np.asarray(y_val), np.asarray(val_pred)))
        if val_rmse < xgb_best_val_rmse:
            xgb_best = params
            xgb_best_val_rmse = val_rmse

    best_xgb = XGBRegressor(**xgb_best, objective="reg:squarederror")
    xgb_metrics = fit_and_score(best_xgb, X_train, y_train, X_test, y_test)
    records.append(
        {
            "stage": "tuned",
            "model": "XGBRegressor",
            "params": xgb_best,
            "metrics": xgb_metrics,
            "validation_rmse": float(xgb_best_val_rmse),
            "rationale": "Gradient boosting is expected to outperform the baseline on nonlinear and interaction-heavy traffic patterns, especially during rush-hour and event-driven anomalies.",
        }
    )

    return records


def save_experiment_summary(records: List[dict], output_path: Path) -> None:
    ordered = []
    for entry in records:
        ordered.append(
            {
                "model": entry["model"],
                "stage": entry["stage"],
                "params": json.dumps(entry["params"], sort_keys=True),
                "mae": entry["metrics"]["mae"],
                "rmse": entry["metrics"]["rmse"],
                "mape": entry["metrics"]["mape"],
                "validation_rmse": entry.get("validation_rmse"),
                "rationale": entry["rationale"],
            }
        )

    df = pd.DataFrame(ordered)
    df.to_csv(output_path, index=False)
