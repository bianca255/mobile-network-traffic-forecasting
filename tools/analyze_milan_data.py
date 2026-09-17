from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream-analyse the Milan telecommunications dataset.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing daily .txt files.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/milan"))
    return parser.parse_args()


def data_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("sms-call-internet-mi-*.txt"))
    if not files:
        raise FileNotFoundError(f"No Milan dataset files found in {data_dir}")
    return files


def read_row(line: str) -> tuple[int, int, float]:
    fields = line.split()
    if len(fields) < 4:
        raise ValueError("Expected at least square_id, timestamp, country_code, and internet traffic")
    return int(fields[0]), int(fields[1]), float(fields[-1])


def stream_totals(files: list[Path]) -> tuple[np.ndarray, int]:
    totals = np.zeros(10001, dtype=np.float64)
    row_count = 0
    with ProcessPoolExecutor(max_workers=4) as executor:
        for file_totals, file_rows in executor.map(_scan_totals_file, files):
            totals += file_totals
            row_count += file_rows
    return totals, row_count


def _scan_totals_file(path: Path) -> tuple[np.ndarray, int]:
    totals = np.zeros(10001, dtype=np.float64)
    row_count = 0
    for chunk in _read_chunks(path):
        valid = chunk[[0, 7]].dropna(subset=[7])
        np.add.at(totals, valid[0].to_numpy(dtype=np.int32), valid[7].to_numpy(dtype=np.float64))
        row_count += len(valid)
    return totals, row_count


def _read_chunks(path: Path):
    dtypes = {0: "int32", 1: "int64", 2: "int32", 3: "float32", 4: "float32", 5: "float32", 6: "float32", 7: "float32"}
    return pd.read_csv(path, sep=r"\s+", header=None, names=list(range(8)), dtype=dtypes, chunksize=500_000, engine="c")


def stream_selected_windows(
    files: list[Path],
    selected_squares: set[int],
    windows: dict[str, tuple[int, int]],
) -> dict[str, dict[int, dict[int, float]]]:
    series = {
        name: {square_id: defaultdict(float) for square_id in selected_squares}
        for name in windows
    }
    relevant_files = [path for path in files if path.name <= "sms-call-internet-mi-2013-12-22.txt"]
    arguments = [(path, selected_squares, windows) for path in relevant_files]
    with ProcessPoolExecutor(max_workers=4) as executor:
        for file_series in executor.map(_scan_selected_file, arguments):
            for name in windows:
                for square_id in selected_squares:
                    series[name][square_id].update(file_series[name][square_id])
    return series


def _scan_selected_file(arguments: tuple[Path, set[int], dict[str, tuple[int, int]]]) -> dict[str, dict[int, dict[int, float]]]:
    path, selected_squares, windows = arguments
    series = {
        name: {square_id: defaultdict(float) for square_id in selected_squares}
        for name in windows
    }
    for chunk in _read_chunks(path):
        chunk = chunk[chunk[0].isin(selected_squares)].copy()
        chunk = chunk.dropna(subset=[0, 1, 7])
        for row in chunk[[0, 1, 7]].itertuples(index=False, name=None):
            square_id, timestamp_ms, internet = int(row[0]), int(row[1]), float(row[2])
            for name, (start_ms, end_ms) in windows.items():
                if start_ms <= timestamp_ms < end_ms:
                    series[name][square_id][timestamp_ms] += internet
    return series


def write_series(output_dir: Path, series: dict[int, dict[int, float]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for square_id, values in series.items():
        output_path = output_dir / f"square_{square_id}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "square_id", "internet_traffic"])
            for timestamp_ms, traffic in sorted(values.items()):
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()
                writer.writerow([timestamp, square_id, f"{traffic:.12g}"])


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    files = data_files(args.data_dir)
    memory_evidence = measure_naive_vs_optimized(files[0])
    started = time.perf_counter()
    totals, row_count = stream_totals(files)
    elapsed_first_pass = time.perf_counter() - started
    top_squares = [int(square_id) for square_id in np.argsort(totals[1:])[::-1][:3] + 1]
    selected_squares = set(top_squares) | {4159, 4556}

    summary = {
        "data_directory": str(args.data_dir.resolve()),
        "file_count": len(files),
        "first_file": files[0].name,
        "last_file": files[-1].name,
        "internet_row_count": row_count,
        "top_squares": [
            {"square_id": square_id, "total_internet_traffic": float(totals[square_id])}
            for square_id in top_squares
        ],
        "required_squares": [4159, 4556],
        "first_pass_seconds": elapsed_first_pass,
        "python_process_rss_bytes": _process_rss_bytes(),
        "memory_evidence": memory_evidence,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "area_totals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["square_id", "total_internet_traffic"])
        for square_id in range(1, len(totals)):
            writer.writerow([square_id, f"{totals[square_id]:.12g}"])

    local_zone = ZoneInfo("Europe/Rome")
    first_period_start = int(datetime(2013, 11, 1, tzinfo=local_zone).timestamp() * 1000)
    first_period_end = int(datetime(2013, 11, 15, tzinfo=local_zone).timestamp() * 1000)
    training_end = int(datetime(2013, 12, 16, tzinfo=local_zone).timestamp() * 1000)
    evaluation_start = training_end
    evaluation_end = int(datetime(2013, 12, 23, tzinfo=local_zone).timestamp() * 1000)
    windows = {
        "first_two_weeks": (first_period_start, first_period_end),
        "training": (first_period_start, training_end),
        "evaluation_week": (evaluation_start, evaluation_end),
    }
    window_series = stream_selected_windows(files, selected_squares, windows)
    for name, series in window_series.items():
        write_series(output_dir / name, series)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return None


def measure_naive_vs_optimized(sample_file: Path) -> dict[str, float]:
    naive = pd.read_csv(sample_file, sep=r"\s+", header=None, names=list(range(8)), engine="c")
    naive_mb = float(naive.memory_usage(deep=True).sum() / 1e6)
    del naive
    gc.collect()

    peak_chunk_mb = 0.0
    for chunk in _read_chunks(sample_file):
        peak_chunk_mb = max(peak_chunk_mb, float(chunk.memory_usage(deep=True).sum() / 1e6))
    return {
        "naive_full_load_mb": naive_mb,
        "chunked_typed_peak_mb": peak_chunk_mb,
        "estimated_reduction_percent": float((1.0 - peak_chunk_mb / naive_mb) * 100.0),
    }


if __name__ == "__main__":
    main()