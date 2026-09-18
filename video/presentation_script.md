# Presentation script (7-10 minutes)

## Slide 1 — Title
Forecasting Mobile Network Traffic with Temporal Feature Engineering and Machine Learning

## Slide 2 — Problem and motivation
Mobile network traffic varies sharply across the day and week. Accurate forecasting matters because operators need to balance capacity, reduce congestion, and maintain service quality during busy periods and event-driven demand spikes.

## Slide 3 — Research objective
This project studies the question of which model best predicts short-horizon mobile traffic under strong daily seasonality and bursty demand conditions. We compare a simple baseline with more flexible nonlinear models to understand the trade-off between interpretability and forecasting power.

## Slide 4 — Data and preprocessing
The project uses the Milan telecommunications activity dataset: 61 daily text files, 65,006,826 Internet records, 10-minute observations, and up to 10,000 geographical squares. Because the raw files are approximately 20 GB, they were processed in bounded chunks with four workers rather than loaded into memory. Internet traffic was aggregated by square and timestamp, then represented as normalized sequences of the previous 24 observations.

## Slide 5 — Time-series diagnostics
The highest-total square was 5161, followed by 5059 and 5259. The first-two-week series had 409 missing 10-minute intervals, so it was regularized and interpolated before diagnostics. For square 5161, lag-1 autocorrelation was 0.976, daily-lag autocorrelation at 144 ten-minute steps was 0.907, and weekly-lag autocorrelation at 1008 steps was 0.933. The ADF p-value was approximately 4.6e-15. These findings support explicit sequence windows and seasonal lag inputs.

## Slide 6 — Model design
Three models are evaluated: Ridge autoregression, LSTM, and GRU. Ridge is a classical lag baseline, while LSTM and GRU explicitly process ordered input sequences through different gated recurrent mechanisms.

## Slide 7 — Hyperparameter tuning
Hyperparameters were selected using a chronological validation split and validation RMSE. Ridge tested regularisation strengths, while LSTM and GRU tested hidden-state size and learning rate using the same 24-step input window. After each candidate set, the best configuration was refit on all pre-evaluation data.

## Slide 8 — Results
The corrected sequential-model evaluation compares Ridge, LSTM, and GRU across squares 5161, 5059, and 5259. The final RMSE values are read from the generated results table, with training time reported alongside accuracy. This tests whether recurrent memory improves over the classical lag baseline rather than comparing overlapping tree ensembles.

## Slide 9 — Critical reflection and limitation
The main failure cases are abrupt traffic spikes and low-traffic intervals, where lagged models cannot anticipate an event and percentage error becomes unstable. The downloaded collection is also missing December 11, and the study does not model cross-area dependence or external events. Future work should add exogenous signals, spatial features, and recurrent or transformer sequence models.

## Slide 10 — Conclusion
The study demonstrates a reproducible memory-aware workflow for real Milan traffic data. Careful aggregation, temporal diagnostics, chronological tuning, and area-level comparison produced a defensible model evaluation. Future work should scale the approach to all 10,000 squares, add cross-area information, and evaluate deeper sequential models.

## Closing
Thank you.
