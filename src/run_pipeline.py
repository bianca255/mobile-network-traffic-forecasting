from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_generation import build_supervised_dataset, generate_mobile_traffic_data, split_train_validation_test
from eda_analysis import compute_time_series_diagnostics, plot_eda_analysis
from report_generator import generate_report_pdf
from train_models import prepare_feature_target, run_experiments, save_experiment_summary


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
RESULTS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)


def plot_forecast(df: pd.DataFrame, predictions: pd.Series, output_path: Path, title: str = "Forecast vs Actual") -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    actual = df["traffic"].tail(len(predictions)).reset_index(drop=True)
    pred = predictions.reset_index(drop=True)
    x = range(len(actual))
    ax.plot(x, actual, label="Actual", linewidth=1.6)
    ax.plot(x, pred, label="Predicted", linewidth=1.2, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Test horizon (hours)")
    ax.set_ylabel("Traffic")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    df = generate_mobile_traffic_data(n_days=180, seed=42)
    supervised = build_supervised_dataset(df)
    train_df, val_df, test_df = split_train_validation_test(supervised, test_days=7, val_days=14)

    X_train, y_train = prepare_feature_target(train_df, target_col="traffic")
    X_val, y_val = prepare_feature_target(val_df, target_col="traffic")
    X_test, y_test = prepare_feature_target(test_df, target_col="traffic")

    records = run_experiments(X_train, y_train, X_val, y_val, X_test, y_test)
    save_experiment_summary(records, RESULTS_DIR / "experiments.csv")

    diagnostics = compute_time_series_diagnostics(df)
    plot_eda_analysis(df, RESULTS_DIR / "eda_diagnostics.png")

    # Choose final winner based on validation RMSE and then fit final evaluation model on train+val
    best = sorted(records, key=lambda x: x["metrics"]["rmse"])[0]
    metrics_summary = {
        "best_model": best["model"],
        "best_mae": best["metrics"]["mae"],
        "best_rmse": best["metrics"]["rmse"],
        "best_mape": best["metrics"]["mape"],
        "linear_mae": next(r["metrics"]["mae"] for r in records if r["model"] == "LinearRegression"),
        "linear_rmse": next(r["metrics"]["rmse"] for r in records if r["model"] == "LinearRegression"),
        "linear_mape": next(r["metrics"]["mape"] for r in records if r["model"] == "LinearRegression"),
        "rf_mae": next(r["metrics"]["mae"] for r in records if r["model"] == "RandomForestRegressor"),
        "rf_rmse": next(r["metrics"]["rmse"] for r in records if r["model"] == "RandomForestRegressor"),
        "rf_mape": next(r["metrics"]["mape"] for r in records if r["model"] == "RandomForestRegressor"),
        "xgb_mae": next(r["metrics"]["mae"] for r in records if r["model"] == "XGBRegressor"),
        "xgb_rmse": next(r["metrics"]["rmse"] for r in records if r["model"] == "XGBRegressor"),
        "xgb_mape": next(r["metrics"]["mape"] for r in records if r["model"] == "XGBRegressor"),
        "diagnostics": diagnostics,
        "experiments": records,
    }

    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # Plot the best performing model forecast
    best_model = best["model"]
    best_model_name = "LinearRegression"
    if best_model == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor
        best_model_name = RandomForestRegressor(**next(r["params"] for r in records if r["model"] == best_model))
    elif best_model == "XGBRegressor":
        from xgboost import XGBRegressor
        best_model_name = XGBRegressor(**next(r["params"] for r in records if r["model"] == best_model), objective="reg:squarederror")
    else:
        from sklearn.linear_model import LinearRegression
        best_model_name = LinearRegression()

    model = best_model_name.fit(pd.concat([X_train, X_val], axis=0), pd.concat([y_train, y_val], axis=0))
    predictions = model.predict(X_test)
    plot_forecast(test_df, pd.Series(predictions), RESULTS_DIR / "forecast_plot.png", title=f"{best_model} Forecast on Test Horizon")

    generate_report_pdf(DOCS_DIR / "report.pdf", metrics_summary)

    print("Pipeline completed successfully.")
    print(json.dumps({
        "best_model": best_model,
        "rmse": round(best["metrics"]["rmse"], 3),
        "mae": round(best["metrics"]["mae"], 3),
        "mape": round(best["metrics"]["mape"], 3),
    }, indent=2))


if __name__ == "__main__":
    main()
