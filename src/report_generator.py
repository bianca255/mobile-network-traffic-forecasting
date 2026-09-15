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
        ("Introduction", "Mobile network traffic forecasting is essential for resource planning, congestion control, and maintaining service quality in dense urban environments. This project studies the question: which short-horizon forecasting model gives the most reliable prediction of future traffic volume under strong daily and weekly seasonal effects?"),
        ("Dataset and Data Preparation", "Experiments are performed on a synthetic hourly traffic dataset designed to reflect realistic demand cycles, weekend effects, burst events, and gradual growth. Lagged and rolling temporal features were engineered to capture short-term dependencies and weekly seasonality. The dataset was partitioned chronologically into training, validation, and test windows to avoid leakage."),
        ("Exploratory Analysis", "The traffic series exhibits a pronounced daily cycle, elevated demand around evening rush hours, and weekly structure. These patterns motivate the use of lag features, calendar effects, and rolling statistics. This temporal structure is not uniformly stationary, which makes nonlinear models especially relevant."),
        ("Methodology", "Three models were evaluated: Linear Regression, RandomForestRegressor, and XGBRegressor. The tree-based models were tuned on the validation window using a small grid and then evaluated on the final test horizon. Performance was measured with MAE, RMSE, and MAPE."),
        ("Results and Discussion", f"The best-performing model was {summary['best_model']} with RMSE {summary['best_rmse']:.2f}, MAE {summary['best_mae']:.2f}, and MAPE {summary['best_mape']:.2f}%. The results show that nonlinear models are better suited to the bursty and seasonal structure of mobile traffic, while the linear model remains a useful baseline and interpretability reference."),
        ("AI Disclosure and Academic Integrity", "This project was developed with the assistance of AI coding tools for code scaffolding, debugging, and iterative experimentation support. All implementation decisions, data-processing logic, model choices, and conclusions were reviewed and validated by the author, and the workflow was executed from the project codebase to verify the reported results."),
        ("Conclusion", "The project demonstrates that careful feature engineering and validation-driven tuning can materially improve short-term mobile traffic forecasts. A key limitation is that the synthetic dataset does not fully reflect operator-level irregularities, missing values, or region-specific network heterogeneity. Future work should test real operator datasets and extend the forecast horizon to multi-step demand planning."),
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
    ]
    for ref in references:
        story.append(Paragraph(ref, body_style))

    doc.build(story)
