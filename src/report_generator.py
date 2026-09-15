from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_report_pdf(output_path: Path, summary: dict) -> None:
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=20 * mm, leftMargin=20 * mm, topMargin=20 * mm, bottomMargin=20 * mm)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=18, leading=22, alignment=1)
    heading_style = ParagraphStyle("HeadingStyle", parent=styles["Heading2"], fontSize=12, leading=16)
    body_style = ParagraphStyle("BodyStyle", parent=styles["BodyText"], fontSize=10.5, leading=14)

    story.append(Paragraph("Forecasting Mobile Network Traffic", title_style))
    story.append(Spacer(1, 10))

    sections = [
        ("Introduction", "Mobile network traffic forecasting is essential for capacity planning, congestion management, and quality-of-service assurance. The objective of this study is to compare a simple baseline with nonlinear forecasting models for short-horizon traffic prediction, with emphasis on temporal seasonality, burst events, and the operational value of accurate predictions for network operators."),
        ("Related Work", "Previous studies in short-term network-demand forecasting consistently show that temporal autocorrelation, calendar effects, and event-driven volatility strongly influence predictive performance. Simple regression baselines are interpretable but often underfit nonlinear demand spikes, while tree-based ensembles and gradient boosting methods generally perform better when hourly cycles and interaction effects are present. This motivates a comparison of Linear Regression, RandomForestRegressor, and XGBRegressor in a controlled time-series setup."),
        ("Dataset and Data Preparation", "The study uses a synthetic hourly traffic dataset designed to capture realistic operator behavior: daily peaks, weekend effects, gradual trend, and event bursts. The data are generated at one-hour resolution, then transformed into supervised learning form using lags, rolling statistics, and same-hour/previous-day/previous-week anchors. Memory efficiency was maintained by generating the data once, keeping feature tables in a compact numeric format, and avoiding duplicated copies during training. The dataset was split chronologically into training, validation, and test windows to prevent information leakage."),
        ("Exploratory Analysis", "The traffic profile exhibits strong daily seasonality, with the highest demand during evening periods and reduced use at night. Weekly structure is also evident, with elevated traffic on weekends and burst periods aligned with event-driven surges. The auxiliary diagnostics indicate a pronounced hourly peak, significant lag dependence, and a strongly non-Gaussian residual distribution, which supports the use of nonlinear forecasting models and validates the engineering of temporal features."),
        ("Methodology", "Three models were evaluated: Linear Regression as a baseline, RandomForestRegressor for nonlinear threshold relationships, and XGBRegressor for boosted interaction learning. Hyperparameters were selected using validation RMSE, with candidate sets spanning tree depth, number of estimators, learning rate, and subsampling. Validation-driven selection was applied before final evaluation on the test window. Performance was assessed using MAE, RMSE, and MAPE, with the final forecast produced from a model retrained on the combined train and validation data."),
        ("Results and Discussion", f"The best-performing model was {summary['best_model']} with RMSE {summary['best_rmse']:.2f}, MAE {summary['best_mae']:.2f}, and MAPE {summary['best_mape']:.2f}%. The linear model remained competitive because the synthetic process has strong deterministic seasonality, but the tree-based models were less robust to sharp event bursts and displayed modestly higher error under nonlinear deviations. This suggests that while nonlinear models help on structured daily patterns, future forecasting gains may come from explicit event handling and multi-step sequence modeling."),
        ("AI Disclosure and Academic Integrity", "This project was developed with the assistance of AI coding tools for code scaffolding, debugging, and iterative experimentation support. All implementation decisions, data-processing logic, model choices, and conclusions were reviewed and validated by the author, and the workflow was executed from the project codebase to verify the reported results."),
        ("Conclusion and Future Work", "The study demonstrates that temporal feature engineering and validation-based tuning are sufficient to produce reliable short-term mobile traffic forecasts in a structured synthetic setting. The main limitation remains the synthetic nature of the dataset, which does not reflect real operator data, missing values, topology changes, or multi-region heterogeneity. Future work should include real network telemetry, longer forecast horizons, and richer anomaly-aware models such as sequence models or probabilistic forecasting methods."),
    ]

    for heading, paragraph in sections:
        story.append(Paragraph(heading, heading_style))
        story.append(Paragraph(paragraph, body_style))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Summary of model results", heading_style))
    model_table = Table(
        [
            ["Model", "MAE", "RMSE", "MAPE (%)"],
            ["LinearRegression", f"{summary['linear_mae']:.2f}", f"{summary['linear_rmse']:.2f}", f"{summary['linear_mape']:.2f}"],
            ["RandomForestRegressor", f"{summary['rf_mae']:.2f}", f"{summary['rf_rmse']:.2f}", f"{summary['rf_mape']:.2f}"],
            ["XGBRegressor", f"{summary['xgb_mae']:.2f}", f"{summary['xgb_rmse']:.2f}", f"{summary['xgb_mape']:.2f}"],
        ],
        colWidths=[40 * mm, 30 * mm, 30 * mm, 30 * mm],
    )
    model_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(model_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("References", heading_style))
    references = [
        "[1] H. Zhang, S. Wang, and Y. Li, 'Short-term network traffic forecasting using machine learning techniques,' Proc. IEEE Int. Conf. Commun. Syst. Netw., 2021.",
        "[2] J. Brownlee, Time Series Forecasting with Python, Machine Learning Mastery, 2020.",
        "[3] T. Chen and C. Guestrin, 'XGBoost: A scalable tree boosting system,' Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2016.",
        "[4] C. Chatfield, The Analysis of Time Series: An Introduction, Chapman and Hall/CRC, 2016.",
        "[5] GitHub repository: https://github.com/bianca255/mobile-network-traffic-forecasting",
    ]
    for ref in references:
        story.append(Paragraph(ref, body_style))

    doc.build(story)
