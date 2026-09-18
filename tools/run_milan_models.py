from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

SEED = 42
DEVICE = torch.device("cpu")


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape": float(np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1e-8))) * 100),
    }


def make_windows(values: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    values = values.astype(np.float32)
    x = np.stack([values[index:index + sequence_length] for index in range(len(values) - sequence_length)])
    y = values[sequence_length:]
    return x[:, :, None], y


class RecurrentRegressor(nn.Module):
    def __init__(self, cell: str, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        recurrent = nn.LSTM if cell == "LSTM" else nn.GRU
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent = recurrent(1, hidden_size, num_layers=num_layers, batch_first=True, dropout=effective_dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.recurrent(sequence)
        return self.output(outputs[:, -1, :]).squeeze(-1)


def fit_recurrent(
    cell: str,
    params: dict,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[RecurrentRegressor, float]:
    model = RecurrentRegressor(cell, params["hidden_size"], params["num_layers"], params["dropout"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_function = nn.MSELoss()
    x_tensor = torch.from_numpy(x_fit).to(DEVICE)
    y_tensor = torch.from_numpy(y_fit).to(DEVICE)
    model.train()
    for _ in range(params["epochs"]):
        permutation = torch.randperm(len(x_tensor))
        for start in range(0, len(x_tensor), params["batch_size"]):
            batch = permutation[start:start + params["batch_size"]]
            optimizer.zero_grad()
            loss = loss_function(model(x_tensor[batch]), y_tensor[batch])
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        validation_prediction = model(torch.from_numpy(x_validation).to(DEVICE)).cpu().numpy()
    return model, metrics(y_validation, validation_prediction)["rmse"]


def predict_recurrent(model: RecurrentRegressor, x_values: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x_values).to(DEVICE)).cpu().numpy()


def recurrent_candidates() -> dict[str, list[dict]]:
    base = {"sequence_length": 24, "num_layers": 1, "dropout": 0.0, "batch_size": 64, "epochs": 12}
    return {
        "LSTM": [
            {**base, "hidden_size": 16, "learning_rate": 0.001},
            {**base, "hidden_size": 32, "learning_rate": 0.001},
        ],
        "GRU": [
            {**base, "hidden_size": 16, "learning_rate": 0.001},
            {**base, "hidden_size": 32, "learning_rate": 0.001},
        ],
    }


def run_area(area: int, input_dir: Path, output_dir: Path) -> tuple[list[dict], list[dict]]:
    seed_everything()
    training = pd.read_csv(input_dir / "training" / f"square_{area}.csv", parse_dates=["timestamp"])
    evaluation = pd.read_csv(input_dir / "evaluation_week" / f"square_{area}.csv", parse_dates=["timestamp"])
    combined = pd.concat([training, evaluation], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    values = combined["internet_traffic"].to_numpy(dtype=np.float32)
    evaluation_start = len(training)
    validation_start = evaluation_start - 7 * 144
    train_values = values[:evaluation_start]
    mean = float(train_values.mean())
    standard_deviation = float(max(train_values.std(), 1e-6))
    normalized = (values - mean) / standard_deviation
    sequence_length = 24
    x_all, y_all = make_windows(normalized, sequence_length)
    target_indices = np.arange(sequence_length, len(values))
    fit_mask = target_indices < validation_start
    validation_mask = (target_indices >= validation_start) & (target_indices < evaluation_start)
    evaluation_mask = target_indices >= evaluation_start
    records: list[dict] = []
    tuning_records: list[dict] = []
    baseline_candidates = [{"alpha": 1.0}, {"alpha": 10.0}, {"alpha": 100.0}]
    baseline_x = x_all.reshape(len(x_all), -1)
    baseline_best: tuple[float, dict] | None = None
    for candidate_index, params in enumerate(baseline_candidates, start=1):
        candidate = make_pipeline(StandardScaler(), Ridge(alpha=params["alpha"]))
        candidate.fit(baseline_x[fit_mask], y_all[fit_mask])
        prediction = candidate.predict(baseline_x[validation_mask])
        validation_metrics = metrics(y_all[validation_mask], prediction)
        score = validation_metrics["rmse"]
        tuning_records.append({"area": area, "model": "RidgeAR", "candidate": candidate_index, "parameters": params, "validation_mae": validation_metrics["mae"], "validation_rmse": score, "validation_mape": validation_metrics["mape"], "selection_reason": "Select lowest validation RMSE."})
        if baseline_best is None or score < baseline_best[0]:
            baseline_best = (score, params)
    baseline_params = baseline_best[1]
    baseline_model = make_pipeline(StandardScaler(), Ridge(alpha=baseline_params["alpha"]))
    train_start = time.perf_counter()
    baseline_model.fit(baseline_x[target_indices < evaluation_start], y_all[target_indices < evaluation_start])
    baseline_training_seconds = time.perf_counter() - train_start
    predict_start = time.perf_counter()
    baseline_prediction = baseline_model.predict(baseline_x[evaluation_mask])
    baseline_execution_seconds = time.perf_counter() - predict_start
    records.append(_record(area, "RidgeAR", baseline_params, baseline_best[0], y_all[evaluation_mask], baseline_prediction, baseline_training_seconds, baseline_execution_seconds, mean, standard_deviation))
    for model_name, candidates in recurrent_candidates().items():
        best: tuple[float, dict] | None = None
        for candidate_index, params in enumerate(candidates, start=1):
            candidate_model, score = fit_recurrent(model_name, params, x_all[fit_mask], y_all[fit_mask], x_all[validation_mask], y_all[validation_mask])
            tuning_records.append({"area": area, "model": model_name, "candidate": candidate_index, "parameters": params, "validation_mae": None, "validation_rmse": score, "validation_mape": None, "selection_reason": "Select lowest validation RMSE."})
            if best is None or score < best[0]:
                best = (score, params)
        selected_params = best[1]
        final_params = {**selected_params, "epochs": selected_params["epochs"] * 2}
        train_start = time.perf_counter()
        final_model, _ = fit_recurrent(model_name, final_params, x_all[target_indices < evaluation_start], y_all[target_indices < evaluation_start], x_all[evaluation_mask], y_all[evaluation_mask])
        training_seconds = time.perf_counter() - train_start
        predict_start = time.perf_counter()
        prediction = predict_recurrent(final_model, x_all[evaluation_mask])
        execution_seconds = time.perf_counter() - predict_start
        records.append(_record(area, model_name, final_params, best[0], y_all[evaluation_mask], prediction, training_seconds, execution_seconds, mean, standard_deviation))
    _write_plots(area, combined, records, output_dir / "plots")
    return records, tuning_records


def _record(area: int, model: str, params: dict, validation_rmse: float, actual_normalized: np.ndarray, prediction_normalized: np.ndarray, training_seconds: float, execution_seconds: float, mean: float, standard_deviation: float) -> dict:
    actual = actual_normalized * standard_deviation + mean
    prediction = prediction_normalized * standard_deviation + mean
    result = metrics(actual, prediction)
    result.update({"area": area, "model": model, "parameters": params, "validation_rmse": validation_rmse, "training_seconds": training_seconds, "execution_seconds": execution_seconds, "_prediction": prediction.tolist()})
    return result


def _write_plots(area: int, combined: pd.DataFrame, records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        predictions = np.asarray(record.pop("_prediction", []))
        if len(predictions) == 0:
            continue
        evaluation_timestamps = combined["timestamp"].iloc[-len(predictions):].reset_index(drop=True)
        actual_values = combined["internet_traffic"].iloc[-len(predictions):].to_numpy()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(evaluation_timestamps, actual_values, label="Actual", linewidth=1.1)
        ax.plot(evaluation_timestamps, predictions, label="Predicted", linewidth=1.0)
        ax.set_title(f"Square {area}: {record['model']}, 16-22 December 2013")
        ax.set_ylabel("Internet traffic")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"square_{area}_{record['model']}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/milan"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/milan"))
    args = parser.parse_args()
    summary = json.loads((args.input_dir / "summary.json").read_text(encoding="utf-8"))
    areas = [item["square_id"] for item in summary["top_squares"]]
    all_records, all_tuning = [], []
    for area in areas:
        records, tuning = run_area(area, args.input_dir, args.output_dir)
        all_records.extend(records)
        all_tuning.extend(tuning)
    results = pd.DataFrame(all_records)
    results.to_csv(args.output_dir / "model_results.csv", index=False)
    results.to_json(args.output_dir / "model_results.json", orient="records", indent=2)
    pd.DataFrame(all_tuning).to_csv(args.output_dir / "tuning_experiments.csv", index=False)
    print(results.to_string(index=False))
