# Forecasting Mobile Network Traffic

This project investigates one-step-ahead forecasting of Milan mobile network traffic using the real telecommunications activity dataset from the assignment. The raw data contains approximately two months of 10-minute observations across 10,000 geographical squares. The study compares three forecasting approaches:

- Ridge autoregression as a regularized classical baseline
- LSTM for gated recurrent sequence modeling
- GRU for a compact gated recurrent comparison

The raw dataset is intentionally not committed because it is approximately 20 GB. Place the downloaded daily files named `sms-call-internet-mi-YYYY-MM-DD.txt` in a local directory and pass that directory to the Milan analysis script.

## Project structure

```text
mobile_network_forecast/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data_generation.py
│   ├── eda_analysis.py
│   ├── train_models.py
│   ├── report_generator.py
│   └── run_pipeline.py
├── results/
│   ├── experiments.csv
│   ├── eda_diagnostics.png
│   ├── forecast_plot.png
│   └── summary.json
├── docs/
│   └── report.pdf
└── video/
    └── presentation_script.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Run the full experiment

```bash
python src/run_pipeline.py
```

## Real Milan dataset workflow

```bash
python tools/analyze_milan_data.py --data-dir "C:\\path\\to\\downloaded\\files" --output-dir results/milan
```

The analyzer uses typed, chunked parsing with four bounded workers, records full-load versus chunked memory evidence, calculates Internet traffic only from the Internet column, identifies the top three squares, and extracts the first two weeks, training history, and 16--22 December evaluation week for the required areas. The raw files remain outside the repository.

The Milan workflow generates:

- `results/milan/area_totals.csv` and `summary.json`
- the required first-two-week and evaluation-week area extracts
- distribution, autocorrelation, and comparison figures
- `model_results.csv` and `tuning_experiments.csv`
- nine actual-versus-predicted forecast plots
- the concise PDF report at `docs/milan_report.pdf`

## Notes

- The project is implemented with reproducibility in mind: all random seeds are fixed and chronological splits prevent leakage.
- The sequential experiment was rerun independently with the same seed and reproduced the MAE, RMSE, and MAPE values to at least four decimal places; this confirms that `seed_everything()` controls the stochastic LSTM/GRU training sufficiently for reproducible comparison. Runtime varies with machine load.
- The methods follow an iterative experimentation strategy: every candidate configuration is recorded with validation metrics and a selection rationale in `results/milan/tuning_experiments.csv`.
- Time-series diagnostics include autocorrelation and stationarity checks to justify the forecasting design.
- Memory management uses typed chunked parsing, four bounded workers, compact numeric accumulators, and excludes the approximately 20 GB raw dataset from Git.
- AI-assisted coding tools were used only to support development and debugging; the final implementation, methodology, and conclusions were reviewed and validated by the author in line with academic integrity requirements.
