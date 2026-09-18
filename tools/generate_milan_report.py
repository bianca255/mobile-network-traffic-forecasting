from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/milan"))
    parser.add_argument("--output", type=Path, default=Path("docs/milan_report.pdf"))
    args = parser.parse_args()
    summary = json.loads((args.input_dir / "summary.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((args.input_dir / "eda_diagnostics.json").read_text(encoding="utf-8"))
    results = pd.read_csv(args.input_dir / "model_results.csv")
    tuning = pd.read_csv(args.input_dir / "tuning_experiments.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.2, leading=12)
    heading = ParagraphStyle("Heading", parent=styles["Heading2"], fontSize=12, leading=15)
    story = [Paragraph("Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting", styles["Title"]), Spacer(1, 8)]
    sections = [
        ("Introduction", "This empirical study investigates one-step-ahead Internet traffic forecasting across Milan geographical squares. The research question is how sequential models compare and whether accuracy changes with traffic characteristics across areas."),
        ("Related Work", "The design follows established evidence that cellular traffic contains strong periodicity, short-term autocorrelation, nonlinear peaks, and spatial heterogeneity. Ridge autoregression provides an interpretable classical sequence baseline; LSTM models use gated memory for longer dependencies; and GRU models provide a compact gated recurrent alternative. These models differ in inductive bias, memory mechanism, and computational cost."),
        ("Dataset and Data Preparation", f"The downloaded Milan dataset contains {summary['file_count']} daily text files and {summary['internet_row_count']:,} Internet-activity records across 10,000 squares. The parser uses only the Internet column when it is present and does not substitute call or SMS values. The raw files total approximately 20 GB and are processed with typed chunks and bounded workers. On a sample file, a full untyped load used {summary['memory_evidence']['naive_full_load_mb']:.1f} MB, while the typed chunk peak was {summary['memory_evidence']['chunked_typed_peak_mb']:.1f} MB, an estimated reduction of {summary['memory_evidence']['estimated_reduction_percent']:.1f}%. Data are aggregated by square and 10-minute timestamp, then represented using lagged traffic, rolling statistics, and cyclical time features."),
        ("Exploratory Analysis", f"The distribution of total Internet traffic across the 10,000 squares is strongly right-skewed: most squares contribute relatively modest totals, while a small number of high-traffic squares form a long upper tail. This concentration is operationally important because aggregate forecasting accuracy can hide poor performance in busy locations. The three highest-total squares were {', '.join(str(x['square_id']) for x in summary['top_squares'])}. In the first-two-week comparison, squares 5161, 5059, and 5259 show the largest amplitudes and more pronounced peaks, whereas squares 4159 and 4556 operate at lower levels with visibly different variability. The shared temporal movement suggests common daily demand drivers, but the different amplitudes and spike patterns indicate heterogeneous area characteristics rather than one universal traffic profile.\n\nFor square 5161, the corrected regularly spaced series contains {diagnostics['missing_intervals_before_interpolation']} missing 10-minute intervals before interpolation. Lag-1 autocorrelation was {diagnostics['lag_1_autocorrelation']:.3f}, lag-144 autocorrelation was {diagnostics['lag_144_autocorrelation']:.3f}, and lag-1008 autocorrelation was {diagnostics['lag_1008_autocorrelation']:.3f}. The augmented Dickey-Fuller p-value was {diagnostics['adf_pvalue']:.4g}. The strong short-term, daily, and weekly dependence supports sequence windows and seasonal lag inputs, while the missing intervals justify explicit regularization before autocorrelation analysis."),
        ("Methodology", "For each selected area, observations before 16 December were split chronologically into a fitting period and a final seven-day validation period. The evaluation period was 16--22 December, interpreted in Europe/Rome local time before conversion to Unix milliseconds. All models predict one step ahead from normalized sequences of the previous 24 ten-minute observations. Ridge uses flattened lag windows; LSTM and GRU receive the same windows as three-dimensional tensors and learn recurrent hidden states. Candidate hyperparameters were compared using validation RMSE, then the selected model was refit on all pre-evaluation data. No future evaluation values were used as predictors. Timing was recorded on an Intel Core i7-10810U CPU @ 1.10 GHz machine with 15.78 GB RAM; PyTorch used CPU execution only, with no GPU acceleration."),
        ("Results and Discussion", "The tables below report MAE, RMSE, MAPE, validation RMSE, training time, and prediction time for each of the three highest-traffic areas. The tuning log records every candidate configuration and its validation metrics, so parameter changes are evidence-led rather than arbitrary. Differences across areas are interpreted alongside their temporal plots: regularized lag models provide a strong baseline, while recurrent models can learn nonlinear temporal state but require more training time."),
        ("Failure Analysis and Limitations", "The most difficult periods are abrupt spikes and low-traffic intervals, where percentage error can become unstable and recurrent models cannot know an event before it appears in the history. The missing 11 December source file is a data-coverage limitation. The study also aggregates activity by square and uses only Internet traffic, so it does not model cross-area dependence, outages, or exogenous events."),
        ("Conclusion and Future Work", "The comparison treats accuracy, runtime, and model transparency as joint evidence rather than selecting solely by one error metric. Future work should add the missing day if available, model all 10,000 areas with scalable distributed storage, include SMS/call covariates, and compare deeper recurrent or transformer sequence models with rolling-origin validation."),
    ]
    for title, text in sections:
        story.extend([Paragraph(title, heading), Paragraph(text, body), Spacer(1, 6)])
    story.extend([Paragraph("Selected evidence figures", heading)])
    for image_name, caption in [
        ("area_traffic_distribution.png", "Figure 1. Distribution of total Internet traffic across all geographical squares."),
        ("first_two_weeks_comparison.png", "Figure 2. First-two-week traffic comparison for the three highest areas and squares 4159 and 4556."),
        ("highest_area_diagnostics.png", "Figure 3. Highest-area time series and autocorrelation diagnostics."),
    ]:
        image_path = args.input_dir / image_name
        if image_path.exists():
            story.extend([Image(str(image_path), width=165 * mm, height=58 * mm), Paragraph(caption, body), Spacer(1, 5)])
    story.extend([Paragraph("Forecast plots for all required area-model combinations", heading), _forecast_grid(args.input_dir), Paragraph("Figure 4. Actual versus predicted Internet traffic for all three models across all three highest-traffic areas.", body), Spacer(1, 8)])
    top_table = [["Square", "Total Internet traffic"]] + [[str(item["square_id"]), f"{item['total_internet_traffic']:.3f}"] for item in summary["top_squares"]]
    story.extend([Paragraph("Top geographical areas", heading), _table(top_table), Spacer(1, 8)])
    story.append(Paragraph("Forecasting results by geographical area", heading))
    for area in sorted(results["area"].unique()):
        area_table = [["Model", "MAE", "RMSE", "MAPE", "Train s", "Predict s"]]
        for row in results[results["area"] == area].itertuples():
            area_table.append([row.model, f"{row.mae:.3f}", f"{row.rmse:.3f}", f"{row.mape:.2f}%", f"{row.training_seconds:.3f}", f"{row.execution_seconds:.4f}"])
        story.extend([Paragraph(f"Square {area}", body), _table(area_table), Spacer(1, 5)])
    tuning_table = [["Area", "Model", "Candidate", "Validation RMSE", "Parameters"]]
    for row in tuning.itertuples():
        tuning_table.append([str(row.area), row.model, str(row.candidate), f"{row.validation_rmse:.3f}", str(row.parameters)])
    story.extend([Spacer(1, 8), Paragraph("Iterative tuning log", heading), _table(tuning_table)])
    references = [
        "[1] G. Barlacchi et al., A multi-source dataset of urban life in the city of Milan and the Province of Trentino, Scientific Data, 2015.",
        "[2] Harvard Dataverse, Milan telecommunications dataset, DOI: 10.7910/DVN/EGZHFV.",
        "[3] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning, Springer.",
        "[4] GitHub repository: https://github.com/bianca255/mobile-network-traffic-forecasting",
        "[5] Demo video: to be added after the individual presentation is recorded and uploaded.",
    ]
    story.extend([Spacer(1, 8), Paragraph("References", heading)])
    story.extend(Paragraph(reference, body) for reference in references)
    doc = SimpleDocTemplate(str(args.output), pagesize=A4, rightMargin=17 * mm, leftMargin=17 * mm, topMargin=15 * mm, bottomMargin=15 * mm)
    doc.build(story)


def _table(rows: list[list[str]]) -> Table:
    table = Table(rows, repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")), ("GRID", (0, 0), (-1, -1), 0.35, colors.grey), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 7)]))
    return table


def _forecast_grid(input_dir: Path) -> Table:
    models = ["RidgeAR", "LSTM", "GRU"]
    areas = [5161, 5059, 5259]
    rows = [["Area"] + models]
    for area in areas:
        cells = [Paragraph(f"Square {area}", ParagraphStyle("GridLabel", fontSize=7, leading=8))]
        for model in models:
            image_path = input_dir / "plots" / f"square_{area}_{model}.png"
            cells.append(Image(str(image_path), width=53 * mm, height=25 * mm))
        rows.append(cells)
    table = Table(rows, colWidths=[22 * mm, 55 * mm, 55 * mm, 55 * mm], rowHeights=[8 * mm, 29 * mm, 29 * mm, 29 * mm])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")), ("GRID", (0, 0), (-1, -1), 0.35, colors.grey), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 7), ("ALIGN", (1, 1), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return table


if __name__ == "__main__":
    main()
