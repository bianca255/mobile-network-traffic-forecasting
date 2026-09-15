from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def generate_mobile_traffic_data(n_days: int = 180, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic mobile network traffic dataset."""
    rng = np.random.default_rng(seed)
    total_hours = n_days * 24
    timestamps = pd.date_range(start="2024-01-01", periods=total_hours, freq="h")

    hours = np.arange(total_hours)
    base = 4000.0
    trend = 20.0 + 0.12 * hours
    daily_cycle = 900.0 * np.sin(2 * np.pi * hours / 24.0)
    weekly_cycle = 550.0 * np.sin(2 * np.pi * hours / (24.0 * 7.0))
    monthly_cycle = 300.0 * np.sin(2 * np.pi * hours / (24.0 * 30.0))

    # Stronger evening peaks and weekend shifts
    hour_of_day = np.mod(hours, 24)
    rush_peak = np.maximum(0.0, 1.0 - np.abs(hour_of_day - 20) / 4.0)
    night_low = np.maximum(0.0, 1.0 - np.abs(hour_of_day - 3) / 5.0)
    weekend_signal = np.where(pd.DatetimeIndex(timestamps).dayofweek >= 5, 1.0, 0.0)
    seasonal_signal = 300.0 * rush_peak + 200.0 * weekend_signal - 180.0 * night_low

    # Event bursts to mimic promotions, outages, or social gatherings
    event_mask = np.zeros(total_hours, dtype=float)
    event_times = [24 * 10, 24 * 25, 24 * 48, 24 * 95, 24 * 130, 24 * 154]
    for event_index in event_times:
        event_mask[event_index : event_index + 8] = np.linspace(0.0, 1.0, 8)
        event_mask[event_index + 8 : event_index + 16] = np.linspace(1.0, 0.3, 8)
    event_mask += np.random.default_rng(seed + 1).uniform(0.0, 0.4, total_hours)

    noise = rng.normal(0.0, 60.0, total_hours)
    traffic = base + trend + daily_cycle + weekly_cycle + monthly_cycle + seasonal_signal + event_mask + noise
    traffic = np.clip(traffic, 500.0, None)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "traffic": traffic,
            "hour": hour_of_day,
            "day_of_week": pd.DatetimeIndex(timestamps).dayofweek,
            "weekend": weekend_signal,
            "month": pd.DatetimeIndex(timestamps).month,
            "day_of_year": pd.DatetimeIndex(timestamps).dayofyear,
            "is_event": event_mask > 0.5,
        }
    )

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    return df


def build_supervised_dataset(df: pd.DataFrame, target_col: str = "traffic") -> pd.DataFrame:
    """Prepare lagged and rolling features for supervised forecasting."""
    result = df.copy().sort_values("timestamp").reset_index(drop=True)

    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    for lag in lags:
        result[f"lag_{lag}"] = result[target_col].shift(lag)

    for window in [3, 6, 12, 24, 72]:
        result[f"rolling_mean_{window}"] = result[target_col].shift(1).rolling(window, min_periods=1).mean()
        result[f"rolling_std_{window}"] = result[target_col].shift(1).rolling(window, min_periods=1).std().fillna(0.0)

    result["prev_hour"] = result[target_col].shift(1)
    result["prev_day_same_hour"] = result[target_col].shift(24)
    result["prev_week_same_hour"] = result[target_col].shift(168)

    result = result.dropna().reset_index(drop=True)
    return result


def split_train_validation_test(df: pd.DataFrame, test_days: int = 7, val_days: int = 14) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a chronological split with validation tuning and final test period."""
    total_hours = len(df)
    test_size = test_days * 24
    val_size = val_days * 24

    test_start = total_hours - test_size
    val_start = test_start - val_size

    train_df = df.iloc[:val_start].copy()
    val_df = df.iloc[val_start:test_start].copy()
    test_df = df.iloc[test_start:].copy()
    return train_df, val_df, test_df
