from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


MODEL_NAMES = ("RidgeAR", "RandomForest", "XGBoost")


def make_features(values: pd.Series, lags: tuple[int, ...] = (1, 2, 3, 6, 12, 24, 72, 144)) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame({"traffic": values.astype("float32")})
    for lag in lags:
        frame[f"lag_{lag}"] = frame["traffic"].shift(lag)
    frame["rolling_mean_6"] = frame["traffic"].shift(1).rolling(6).mean()
    frame["rolling_mean_144"] = frame["traffic"].shift(1).rolling(144).mean()
    frame["rolling_std_24"] = frame["traffic"].shift(1).rolling(24).std()
    frame["hour_sin"] = np.sin(2 * np.pi * np.arange(len(frame)) / 144)
    frame["hour_cos"] = np.cos(2 * np.pi * np.arange(len(frame)) / 144)
    frame["day_sin"] = np.sin(2 * np.pi * np.arange(len(frame)) / (144 * 7))
    frame["day_cos"] = np.cos(2 * np.pi * np.arange(len(frame)) / (144 * 7))
    frame = frame.dropna()
    return frame.drop(columns="traffic"), frame["traffic"]


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape": float(np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1e-8))) * 100),
    }


def make_models() -> dict[str, list[tuple[dict, object]]]:
    return {
        "RidgeAR": [
            ({"alpha": 1.0}, make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
            ({"alpha": 10.0}, make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
            ({"alpha": 100.0}, make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ],
        "RandomForest": [
            ({"n_estimators": 120, "max_depth": 12, "min_samples_leaf": 2}, RandomForestRegressor(n_estimators=120, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)),
            ({"n_estimators": 200, "max_depth": 18, "min_samples_leaf": 2}, RandomForestRegressor(n_estimators=200, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ],
        "XGBoost": [
            ({"n_estimators": 250, "max_depth": 5, "learning_rate": 0.05}, XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror", random_state=42, n_jobs=4)),
            ({"n_estimators": 400, "max_depth": 6, "learning_rate": 0.03}, XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror", random_state=42, n_jobs=4)),
        ],
    }


def run_area(area: int, input_dir: Path, output_dir: Path) -> list[dict]:
    training = pd.read_csv(input_dir / "training" / f"square_{area}.csv", parse_dates=["timestamp"])
    evaluation = pd.read_csv(input_dir / "evaluation_week" / f"square_{area}.csv", parse_dates=["timestamp"])
    combined = pd.concat([training, evaluation], ignore_index=True).sort_values("timestamp")
    x_all, y_all = make_features(combined["internet_traffic"])
    eval_start = evaluation["timestamp"].min()
    eval_mask = combined.loc[x_all.index, "timestamp"] >= eval_start
    train_mask = ~eval_mask
    validation_start = training["timestamp"].max() - pd.Timedelta(days=7)
    validation_mask = train_mask & (combined.loc[x_all.index, "timestamp"] >= validation_start)
    fit_mask = train_mask & ~validation_mask
    area_dir = output_dir / "plots"
    area_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for model_name, candidates in make_models().items():
        best = None
        for params, candidate in candidates:
            candidate.fit(x_all.loc[fit_mask], y_all.loc[fit_mask])
            val_prediction = candidate.predict(x_all.loc[validation_mask])
            score = metrics(y_all.loc[validation_mask].to_numpy(), val_prediction)["rmse"]
            if best is None or score < best[0]:
                best = (score, params)
        selected_params = best[1]
        final_model = next(model for params, model in make_models()[model_name] if params == selected_params)
        train_start = time.perf_counter()
        final_model.fit(x_all.loc[train_mask], y_all.loc[train_mask])
        training_seconds = time.perf_counter() - train_start
        predict_start = time.perf_counter()
        prediction = final_model.predict(x_all.loc[eval_mask])
        execution_seconds = time.perf_counter() - predict_start
        actual = y_all.loc[eval_mask].to_numpy()
        result = metrics(actual, prediction)
        result.update({"area": area, "model": model_name, "parameters": selected_params, "validation_rmse": best[0], "training_seconds": training_seconds, "execution_seconds": execution_seconds})
        records.append(result)
        plot_data = combined.loc[x_all.loc[eval_mask].index, ["timestamp"]].copy()
        plot_data["actual"] = actual
        plot_data["predicted"] = prediction
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(plot_data["timestamp"], plot_data["actual"], label="Actual", linewidth=1.1)
        ax.plot(plot_data["timestamp"], plot_data["predicted"], label="Predicted", linewidth=1.0)
        ax.set_title(f"Square {area}: {model_name}, 16-22 December 2013")
        ax.set_ylabel("Internet traffic")
        ax.legend()
        fig.tight_layout()
        fig.savefig(area_dir / f"square_{area}_{model_name}.png", dpi=180)
        plt.close(fig)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/milan"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/milan"))
    args = parser.parse_args()
    summary = json.loads((args.input_dir / "summary.json").read_text(encoding="utf-8"))
    areas = [item["square_id"] for item in summary["top_squares"]]
    records = [record for area in areas for record in run_area(area, args.input_dir, args.output_dir)]
    results = pd.DataFrame(records)
    results.to_csv(args.output_dir / "model_results.csv", index=False)
    results.to_json(args.output_dir / "model_results.json", orient="records", indent=2)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
