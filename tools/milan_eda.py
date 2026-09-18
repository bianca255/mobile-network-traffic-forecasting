from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/milan"))
    args = parser.parse_args()
    summary = json.loads((args.input_dir / "summary.json").read_text(encoding="utf-8"))
    area_totals = pd.read_csv(args.input_dir / "area_totals.csv")
    first_dir = args.input_dir / "first_two_weeks"
    areas = [item["square_id"] for item in summary["top_squares"]] + [4159, 4556]
    first_series = {area: _read_series(first_dir / f"square_{area}.csv") for area in areas}
    totals = pd.Series({area: series["internet_traffic"].sum() for area, series in first_series.items()})
    totals.to_csv(args.input_dir / "selected_area_totals.csv", header=["first_two_week_total"])
    _plot_distribution(area_totals, args.input_dir / "area_traffic_distribution.png")
    _plot_area_series(first_series, args.input_dir / "first_two_weeks_comparison.png")
    highest = summary["top_squares"][0]["square_id"]
    raw_highest_series = _read_series(first_dir / f"square_{highest}.csv").set_index("timestamp")["internet_traffic"]
    full_index = pd.date_range(raw_highest_series.index.min(), raw_highest_series.index.max(), freq="10min")
    missing_count = len(full_index) - len(raw_highest_series)
    highest_series = raw_highest_series.reindex(full_index).interpolate(limit_direction="both")
    diagnostics = {
        "expected_intervals": len(full_index),
        "missing_intervals_before_interpolation": int(missing_count),
        "highest_square": highest,
        "adf_statistic": float(adfuller(highest_series, autolag="AIC")[0]),
        "adf_pvalue": float(adfuller(highest_series, autolag="AIC")[1]),
        "lag_1_autocorrelation": float(highest_series.autocorr(1)),
        "lag_144_autocorrelation": float(highest_series.autocorr(144)),
        "lag_1008_autocorrelation": float(highest_series.autocorr(1008)),
        "mean": float(highest_series.mean()),
        "standard_deviation": float(highest_series.std()),
        "maximum": float(highest_series.max()),
    }
    (args.input_dir / "eda_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    _plot_diagnostics(highest_series, args.input_dir / "highest_area_diagnostics.png")
    print(json.dumps(diagnostics, indent=2))


def _read_series(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
    return frame


def _plot_area_series(series: dict[int, pd.DataFrame], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for area, frame in series.items():
        ax.plot(frame["timestamp"], frame["internet_traffic"], linewidth=0.8, label=f"Square {area}")
    ax.set_title("Internet traffic during the first two weeks")
    ax.set_ylabel("Aggregated Internet traffic")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_distribution(area_totals: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(area_totals["total_internet_traffic"], bins=50, color="#2f6f8f", alpha=0.85)
    ax.set_title("Distribution of total Internet traffic across Milan squares")
    ax.set_xlabel("Total Internet traffic")
    ax.set_ylabel("Number of squares")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_diagnostics(series: pd.Series, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    series.plot(ax=axes[0], linewidth=0.8, title="Highest-traffic square: first two weeks")
    plot_acf(series, lags=288, ax=axes[1], zero=False)
    axes[1].set_title("Autocorrelation through 48 hours")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
