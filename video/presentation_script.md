# Presentation script (7-10 minutes)

## Slide 1 — Title
"Forecasting Mobile Network Traffic with Temporal Feature Engineering and Tree-Based Models"

## Slide 2 — Problem and motivation
Mobile network traffic varies strongly by hour, weekday, and special events. Accurate forecasting supports load balancing, capacity planning, and congestion mitigation. The research question is: which model best captures these nonlinear, highly seasonal patterns?

## Slide 3 — Data and preprocessing
The study uses a synthetic hourly dataset designed to mimic real operator traffic: daily peaks, weekend effects, growth trend, and event spikes. Features include lags, rolling means, calendar effects, and previous-day/previous-week signals. The data are split chronologically into training, validation, and test windows.

## Slide 4 — Methodology
Three models are evaluated: Linear Regression as a baseline, RandomForestRegressor for nonlinear patterns, and XGBoostRegressor for stronger gradient-boosted performance. Hyperparameters are selected using a validation set and a small tuning grid, so the final model is chosen on evidence rather than a single lucky run.

## Slide 5 — Important technical decision
A key decision was to engineer temporal features rather than rely on raw values alone. Lagged traffic, rolling windows, and time-of-day signals are essential because the data show strong self-dependence and daily seasonality. This improved the models significantly compared with using only raw timestamps.

## Slide 6 — Results
The experiments show that the XGBoost model improves on the linear baseline in RMSE and MAPE, while RandomForest remains competitive. The best performance is achieved by the model that captures complex interactions between rush hours, weekly cycles, and burst events.

## Slide 7 — Limitation and failure case
A notable limitation is that the dataset is synthetic and cannot reproduce all irregularities of real mobile network traffic, such as outages, topology changes, and multi-region interference. During sudden spikes, even the tuned models can lag behind the realized demand, which suggests that extreme anomaly handling could be improved with event-aware features or deep sequence models.

## Slide 8 — Conclusion
The project shows that careful preprocessing and validation-driven tuning can produce reliable short-term forecasts for mobile network traffic. Future work can expand the study to real operator data, longer forecast horizons, and probabilistic forecasting models.

## Closing
"Thank you."
