# Forecasting Mobile Network Traffic

This project investigates one-step-ahead forecasting of Milan mobile network traffic using the real telecommunications activity dataset from the assignment. The raw data contains approximately two months of 10-minute observations across 10,000 geographical squares. The study compares three forecasting approaches:

- Linear regression as a simple interpretable baseline
- Random forest regression for nonlinear temporal relationships
- XGBoost for stronger gradient-boosted performance on tabular time-series features

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

The analyzer streams the raw files line by line, records process memory, calculates total Internet traffic per square, identifies the top three squares, and extracts the first two weeks, training history, and 16--22 December evaluation week for the required areas. The raw files remain outside the repository.

This generates:

- a synthetic mobile network traffic dataset
- time-series diagnostics and EDA plots
- feature engineering for lagged temporal signals
- model training and validation-based hyperparameter tuning
- a summary in `results/summary.json`
- an EDA diagnostic figure in `results/eda_diagnostics.png`
- a forecast plot in `results/forecast_plot.png`
- a compact PDF report in `docs/report.pdf`

## Notes

- The project is implemented with reproducibility in mind: all random seeds are fixed.
- Data are generated to resemble realistic mobile traffic, because the assignment requires a complete experimental workflow without relying on a pre-provided dataset.
- The methods follow an iterative experimentation strategy: the baseline is established first, then the tree-based models are tuned on a validation split before final evaluation.
- Time-series diagnostics include autocorrelation assessment, seasonal decomposition, and stationarity checks to justify the forecasting design.
- Memory management is handled via compact numerical representations, chronological splitting, and disciplined feature generation so that the pipeline remains efficient without duplicating the full data array.
- AI-assisted coding tools were used only to support development and debugging; the final implementation, methodology, and conclusions were reviewed and validated by the author in line with academic integrity requirements.
