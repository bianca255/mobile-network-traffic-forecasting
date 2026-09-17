# Presentation script (7-10 minutes)

## Slide 1 — Title
Forecasting Mobile Network Traffic with Temporal Feature Engineering and Machine Learning

## Slide 2 — Problem and motivation
Mobile network traffic varies sharply across the day and week. Accurate forecasting matters because operators need to balance capacity, reduce congestion, and maintain service quality during busy periods and event-driven demand spikes.

## Slide 3 — Research objective
This project studies the question of which model best predicts short-horizon mobile traffic under strong daily seasonality and bursty demand conditions. We compare a simple baseline with more flexible nonlinear models to understand the trade-off between interpretability and forecasting power.

## Slide 4 — Data and preprocessing
The project uses the Milan telecommunications activity dataset: 61 daily text files, 314,412,299 records, 10-minute observations, and up to 10,000 geographical squares. Because the raw files are approximately 20 GB, they were processed in bounded chunks with four workers rather than loaded into memory. Internet traffic was aggregated by square and timestamp, then represented with lag features, rolling statistics, and cyclical time variables.

## Slide 5 — Time-series diagnostics
The highest-total square was 5161, followed by 5059 and 5259. For square 5161, lag-1 autocorrelation was 0.968. The daily and weekly lag correlations were weaker after isolating Internet-only records, while the ADF p-value was approximately 6.4e-17. This supports short-term lag predictors while warning that daily and weekly behavior is not uniform across every activity stream.

## Slide 6 — Model design
Three models are evaluated: Ridge autoregression, Random Forest, and XGBoost. Ridge is a regularized linear lag model and interpretable baseline, while the tree-based models are designed to capture nonlinear relationships and threshold effects that appear during traffic surges.

## Slide 7 — Hyperparameter tuning
Hyperparameters were selected using a chronological validation split and validation RMSE. Ridge tested regularisation strengths, Random Forest tested estimator count, depth, and leaf size, and XGBoost tested estimator count, depth, and learning rate. After each candidate set, the best configuration was refit on all pre-evaluation data.

## Slide 8 — Results
On the December 16–22 evaluation week, XGBoost achieved the best RMSE for square 5161 at 136.31. Ridge achieved the best RMSE for square 5059 at 111.82 and square 5259 at 105.96. This shows that model performance varies with area characteristics and that model complexity is not automatically better.

## Slide 9 — Critical reflection and limitation
The main failure cases are abrupt traffic spikes and low-traffic intervals, where lagged models cannot anticipate an event and percentage error becomes unstable. The downloaded collection is also missing December 11, and the study does not model cross-area dependence or external events. Future work should add exogenous signals, spatial features, and recurrent or transformer sequence models.

## Slide 10 — Conclusion
The study demonstrates a reproducible memory-aware workflow for real Milan traffic data. Careful aggregation, temporal diagnostics, chronological tuning, and area-level comparison produced a defensible model evaluation. Future work should scale the approach to all 10,000 squares, add cross-area information, and evaluate deeper sequential models.

## Closing
Thank you.
