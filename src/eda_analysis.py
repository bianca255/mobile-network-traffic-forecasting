from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller


def compute_time_series_diagnostics(df: pd.DataFrame) -> dict:
    """Compute stationarity, autocorrelation, and seasonal diagnostics for the traffic series."""
    ts = df.set_index("timestamp")["traffic"].sort_index().asfreq("h")
    adf_stat, adf_pvalue, *_ = adfuller(ts, autolag="AIC")
    lag_1 = acf(ts, nlags=1, fft=True)[1]
    lag_24 = acf(ts, nlags=24, fft=True)[24]
    lag_168 = acf(ts, nlags=168, fft=True)[168]
    decomp = seasonal_decompose(ts, model="additive", period=24, extrapolate_trend="freq")

    daily_profile = ts.groupby(ts.index.hour).mean()
    weekly_profile = ts.groupby(ts.index.dayofweek).mean()
    summary = {
        "adf_statistic": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "lag_1_autocorrelation": float(lag_1),
        "lag_24_autocorrelation": float(lag_24),
        "lag_168_autocorrelation": float(lag_168),
        "hourly_peak_hour": int(daily_profile.idxmax()),
        "hourly_peak_traffic": float(daily_profile.max()),
        "weekday_peak_day": int(weekly_profile.idxmax()),
        "weekday_peak_traffic": float(weekly_profile.max()),
        "event_count": int(df["is_event"].sum()),
        "mean_traffic": float(ts.mean()),
        "std_traffic": float(ts.std()),
        "max_traffic": float(ts.max()),
        "min_traffic": float(ts.min()),
        "seasonal_strength": float(np.std(decomp.seasonal.dropna()) / max(np.std(ts), 1e-8)),
    }
    return summary


def plot_eda_analysis(df: pd.DataFrame, output_path: Path) -> None:
    """Generate a multi-panel diagnostic plot for the traffic time series."""
    ts = df.set_index("timestamp")["traffic"].sort_index().asfreq("h")
    daily_profile = ts.groupby(ts.index.hour).mean()
    weekly_profile = ts.groupby(ts.index.dayofweek).mean()
    seasonal = seasonal_decompose(ts, model="additive", period=24, extrapolate_trend="freq")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    axes[0].plot(ts.index, ts.values, color="#1f77b4", linewidth=1.3)
    axes[0].set_title("Hourly traffic time series")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Traffic")

    axes[1].plot(daily_profile.index, daily_profile.values, color="#ff7f0e", linewidth=2)
    axes[1].set_title("Average traffic by hour of day")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Traffic")

    axes[2].bar(weekly_profile.index, weekly_profile.values, color="#2ca02c")
    axes[2].set_title("Average traffic by day of week")
    axes[2].set_xlabel("Day of week")
    axes[2].set_ylabel("Traffic")

    plot_acf(ts, lags=48, ax=axes[3], color="#9467bd")
    axes[3].set_title("Autocorrelation function (ACF)")

    seasonal.observed.plot(ax=axes[4], color="#17becf", linewidth=1.2)
    seasonal.seasonal.plot(ax=axes[4], color="#bcbd22", linewidth=1.0)
    axes[4].set_title("Observed vs seasonal component")
    axes[4].set_ylabel("Traffic")

    residual = seasonal.resid.dropna()
    axes[5].hist(residual, bins=30, color="#d62728", alpha=0.7)
    axes[5].set_title("Residual distribution")
    axes[5].set_xlabel("Residual")
    axes[5].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
