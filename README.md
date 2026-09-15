# Forecasting Mobile Network Traffic

This project investigates short-horizon forecasting of mobile network traffic using a realistic synthetic hourly dataset that captures daily seasonality, weekly cycles, growth trends, and anomalous demand spikes. The study compares three forecasting approaches:

- Linear regression as a simple interpretable baseline
- Random forest regression for nonlinear temporal relationships
- XGBoost for stronger gradient-boosted performance on tabular time-series features

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
