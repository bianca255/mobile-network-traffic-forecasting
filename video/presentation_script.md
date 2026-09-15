# Presentation script (7-10 minutes)

## Slide 1 — Title
Forecasting Mobile Network Traffic with Temporal Feature Engineering and Machine Learning

## Slide 2 — Problem and motivation
Mobile network traffic varies sharply across the day and week. Accurate forecasting matters because operators need to balance capacity, reduce congestion, and maintain service quality during busy periods and event-driven demand spikes.

## Slide 3 — Research objective
This project studies the question of which model best predicts short-horizon mobile traffic under strong daily seasonality and bursty demand conditions. We compare a simple baseline with more flexible nonlinear models to understand the trade-off between interpretability and forecasting power.

## Slide 4 — Data and preprocessing
The project uses a synthetic hourly dataset designed to resemble realistic operator behavior. It includes daily peaks, weekend effects, a slow growth trend, and burst periods that mimic promotions or unusual load conditions. We engineered lag features, rolling statistics, and calendar variables so the models can learn temporal dependence instead of relying only on raw timestamps.

## Slide 5 — Time-series diagnostics
The time-series analysis shows clear daily structure and strong autocorrelation. The ACF remains elevated at small lags, and an hourly traffic profile indicates the strongest demand occurs in evening periods. This matters because it confirms the presence of seasonality and supports the choice of lag-based predictors and calendar features.

## Slide 6 — Model design
Three models are evaluated: Linear Regression, RandomForestRegressor, and XGBRegressor. The linear model is a strong baseline and easy to interpret, while the tree-based models are designed to capture nonlinear relationships and threshold effects that appear during traffic surges.

## Slide 7 — Hyperparameter tuning
Hyperparameters were selected using a validation split and validation RMSE, rather than by guessing. The tuning process was intentionally small and disciplined, focusing on the most relevant parameters such as tree depth, number of estimators, and learning rate.

## Slide 8 — Results
The final evaluation showed that the linear model achieved the best test performance in this synthetic setup. This is an important result: when the series is dominated by deterministic daily and weekly structure, a well-designed linear model can outperform more complex models. It also shows that the forecasting challenge is not simply about model complexity; it is about matching the model to the temporal structure of the data.

## Slide 9 — Critical reflection and limitation
A major limitation is that the dataset is synthetic, so it does not include network outages, missing data, topology changes, or multi-region heterogeneity. During abrupt demand swings, the models can still lag the actual change. This suggests that real-world forecasting would benefit from anomaly-aware features, richer temporal encoding, or sequence-based models.

## Slide 10 — Conclusion
The study demonstrates that careful preprocessing, validation-based tuning, and time-series diagnostics are enough to build a credible forecasting pipeline. Future work should extend this to real operator data, multi-step forecasting, and more advanced methods that explicitly model irregular events and nonlinear dynamics.

## Closing
Thank you.
