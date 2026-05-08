from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import html
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SWEEP_CSV = "cadica_threshold_sweep.csv"
SWEEP_SUMMARY_JSON = "cadica_threshold_sweep_summary.json"
CADICA_SUMMARY_JSON = "cadica_summary.json"
MULTIVIEW_PATIENT_ROWS_CSV = "cadica_multiview_patient_rows.csv"
MULTIVIEW_SIDE_ROWS_CSV = "cadica_multiview_side_rows.csv"

CONFIG_COLUMNS = (
    "frame_min_degree",
    "box_margin_px",
    "video_prediction_source",
    "multiview_min_score",
)
DISPLAY_METRICS = (
    "total_evaluated",
    "TP",
    "FP",
    "TN",
    "FN",
    "precision",
    "recall",
    "specificity",
    "F1",
    "balanced_accuracy",
)
STAGES = (
    ("frame", "Frame level"),
    ("video", "Temporal/video level"),
    ("patient", "Patient aggregation"),
    ("multiview_patient", "Multi-view / patient level"),
    ("multiview_side", "Multi-view / coronary-side level"),
)


class CadicaReportError(RuntimeError):
    """Raised when a CADICA benchmark report cannot be generated."""


@dataclass(frozen=True, slots=True)
class CadicaReportArtifacts:
    output_root: Path
    tables: dict[str, Path]
    plots: dict[str, Path]
    reports: dict[str, Path]
    warnings: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a final supervised CADICA benchmark report from existing benchmark outputs.",
    )
    parser.add_argument("--benchmark-root", required=True, help="Directory containing CADICA benchmark/sweep outputs.")
    parser.add_argument(
        "--output-root",
        help="Directory where final report outputs are written. Defaults to <benchmark-root>/final_report.",
    )
    parser.add_argument("--top-k", type=int, default=25, help="Number of top operating points to show.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        artifacts = run_cadica_report(
            benchmark_root=args.benchmark_root,
            output_root=args.output_root,
            top_k=args.top_k,
        )
    except (CadicaReportError, FileNotFoundError, NotADirectoryError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"CADICA report failed: {exc}", file=sys.stderr)
        return 2

    for warning in artifacts.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    print(f"CADICA benchmark report: {artifacts.reports['markdown']}")
    print(f"HTML report: {artifacts.reports['html']}")
    print(f"Summary JSON: {artifacts.reports['summary_json']}")
    return 0


def run_cadica_report(
    *,
    benchmark_root: str | Path,
    output_root: str | Path | None = None,
    top_k: int = 25,
) -> CadicaReportArtifacts:
    if top_k <= 0:
        raise ValueError("--top-k must be positive.")

    resolved_benchmark_root = Path(benchmark_root)
    if not resolved_benchmark_root.exists():
        raise FileNotFoundError(f"CADICA benchmark root does not exist: {resolved_benchmark_root}")
    if not resolved_benchmark_root.is_dir():
        raise NotADirectoryError(f"CADICA benchmark root is not a directory: {resolved_benchmark_root}")

    resolved_output_root = Path(output_root) if output_root is not None else resolved_benchmark_root / "final_report"
    tables_root = resolved_output_root / "tables"
    plots_root = resolved_output_root / "plots"
    tables_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    sweep_csv = resolved_benchmark_root / SWEEP_CSV
    sweep_summary_path = resolved_benchmark_root / SWEEP_SUMMARY_JSON
    cadica_summary_path = resolved_benchmark_root / CADICA_SUMMARY_JSON
    if not sweep_csv.is_file():
        raise FileNotFoundError(
            f"CADICA threshold sweep CSV was not found: {sweep_csv}. "
            "Run scripts/run_cadica_benchmark_sweep.py first."
        )

    sweep_rows = _read_csv(sweep_csv)
    if not sweep_rows:
        raise CadicaReportError(f"CADICA threshold sweep CSV is empty: {sweep_csv}")

    sweep_summary = _read_json(sweep_summary_path) if sweep_summary_path.is_file() else {}
    cadica_summary = _read_json(cadica_summary_path) if cadica_summary_path.is_file() else {}
    warnings = _build_warnings(sweep_rows, sweep_summary, resolved_benchmark_root)

    best_by_stage = _best_by_stage(sweep_rows)
    primary_prefix = "multiview_patient" if best_by_stage.get("multiview_patient") is not None else "patient"
    best_primary = best_by_stage.get(primary_prefix)
    if best_primary is None:
        raise CadicaReportError("No evaluated CADICA patient or multi-view patient metrics were found.")

    top_primary = _rank_rows(sweep_rows, primary_prefix)[:top_k]
    pareto_primary = _pareto_rows(_rank_rows(sweep_rows, primary_prefix), primary_prefix)
    selected_patient_rows = _selected_multiview_rows(
        resolved_benchmark_root / MULTIVIEW_PATIENT_ROWS_CSV,
        best_primary,
    )
    selected_side_rows = _selected_multiview_rows(
        resolved_benchmark_root / MULTIVIEW_SIDE_ROWS_CSV,
        best_by_stage.get("multiview_side") or best_primary,
    )

    table_paths = {
        "all_sweep_metrics": tables_root / "cadica_threshold_sweep_metrics.csv",
        "best_operating_points": tables_root / "best_operating_points.csv",
        "top_primary_operating_points": tables_root / "top_primary_operating_points.csv",
        "pareto_primary_operating_points": tables_root / "pareto_primary_operating_points.csv",
    }
    _write_csv(table_paths["all_sweep_metrics"], sweep_rows, fieldnames=list(sweep_rows[0]))
    _write_csv(table_paths["best_operating_points"], _stage_table_rows(best_by_stage), fieldnames=_stage_table_columns())
    _write_csv(table_paths["top_primary_operating_points"], top_primary, fieldnames=list(sweep_rows[0]))
    _write_csv(table_paths["pareto_primary_operating_points"], pareto_primary, fieldnames=list(sweep_rows[0]))

    if selected_patient_rows:
        table_paths["selected_multiview_patient_rows"] = tables_root / "selected_multiview_patient_rows.csv"
        _write_csv(
            table_paths["selected_multiview_patient_rows"],
            selected_patient_rows,
            fieldnames=list(selected_patient_rows[0]),
        )
    if selected_side_rows:
        table_paths["selected_multiview_side_rows"] = tables_root / "selected_multiview_side_rows.csv"
        _write_csv(
            table_paths["selected_multiview_side_rows"],
            selected_side_rows,
            fieldnames=list(selected_side_rows[0]),
        )

    plot_paths = {
        "precision_recall": plots_root / "01_precision_recall_scatter.png",
        "top_primary_f1": plots_root / "02_top_primary_f1.png",
        "stage_comparison": plots_root / "03_stage_f1_comparison.png",
        "frame_heatmap": plots_root / "04_frame_threshold_heatmap.png",
        "multiview_threshold_curve": plots_root / "05_multiview_threshold_curve.png",
    }
    plot_precision_recall(sweep_rows, primary_prefix, best_primary, pareto_primary, plot_paths["precision_recall"])
    plot_top_f1(top_primary, primary_prefix, plot_paths["top_primary_f1"])
    plot_stage_comparison(best_by_stage, plot_paths["stage_comparison"])
    plot_frame_heatmap(sweep_rows, plot_paths["frame_heatmap"])
    plot_multiview_threshold_curve(sweep_rows, plot_paths["multiview_threshold_curve"])

    summary_payload = build_summary_payload(
        benchmark_root=resolved_benchmark_root,
        output_root=resolved_output_root,
        top_k=top_k,
        sweep_rows=sweep_rows,
        sweep_summary=sweep_summary,
        cadica_summary=cadica_summary,
        best_by_stage=best_by_stage,
        primary_prefix=primary_prefix,
        best_primary=best_primary,
        table_paths=table_paths,
        plot_paths=plot_paths,
        warnings=warnings,
    )

    reports = {
        "markdown": resolved_output_root / "report.md",
        "html": resolved_output_root / "report.html",
        "summary_json": resolved_output_root / "summary.json",
    }
    write_markdown_report(
        reports["markdown"],
        summary=summary_payload,
        best_by_stage=best_by_stage,
        top_primary=top_primary[: min(10, top_k)],
        pareto_primary=pareto_primary[:10],
        primary_prefix=primary_prefix,
        table_paths=table_paths,
        plot_paths=plot_paths,
        output_root=resolved_output_root,
    )
    write_html_report(
        reports["html"],
        summary=summary_payload,
        best_by_stage=best_by_stage,
        top_primary=top_primary[: min(10, top_k)],
        pareto_primary=pareto_primary[:10],
        primary_prefix=primary_prefix,
        table_paths=table_paths,
        plot_paths=plot_paths,
        output_root=resolved_output_root,
    )
    _write_json(reports["summary_json"], summary_payload)

    return CadicaReportArtifacts(
        output_root=resolved_output_root,
        tables=table_paths,
        plots=plot_paths,
        reports=reports,
        warnings=warnings,
    )


def plot_precision_recall(
    rows: list[dict[str, Any]],
    prefix: str,
    best_row: dict[str, Any],
    pareto_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    points = [row for row in rows if _metric(row, prefix, "precision") is not None and _metric(row, prefix, "recall") is not None]
    if not points:
        _plot_empty(output_path, "No Precision/Recall Data")
        return

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.scatter(
        [_metric(row, prefix, "recall") for row in points],
        [_metric(row, prefix, "precision") for row in points],
        s=28,
        alpha=0.42,
        color="#4c78a8",
        label="Operating points",
    )
    if pareto_rows:
        ax.scatter(
            [_metric(row, prefix, "recall") for row in pareto_rows],
            [_metric(row, prefix, "precision") for row in pareto_rows],
            s=54,
            alpha=0.85,
            color="#f58518",
            label="Pareto front",
        )
    ax.scatter(
        [_metric(best_row, prefix, "recall")],
        [_metric(best_row, prefix, "precision")],
        s=96,
        marker="*",
        color="#e45756",
        label="Best F1",
        zorder=5,
    )
    for row in _rank_rows(points, prefix)[:5]:
        recall = _metric(row, prefix, "recall")
        precision = _metric(row, prefix, "precision")
        if recall is not None and precision is not None:
            ax.annotate(_compact_config_label(row), (recall, precision), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_title(f"{_stage_title(prefix)} Precision/Recall Tradeoff")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_top_f1(rows: list[dict[str, Any]], prefix: str, output_path: Path) -> None:
    if not rows:
        _plot_empty(output_path, "No Top Operating Points")
        return
    labels = [_compact_config_label(row) for row in rows][::-1]
    values = [_metric(row, prefix, "F1") or 0.0 for row in rows][::-1]
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.34 * len(rows))))
    ax.barh(labels, values, color="#4c78a8")
    for index, value in enumerate(values):
        precision = _metric(rows[::-1][index], prefix, "precision")
        recall = _metric(rows[::-1][index], prefix, "recall")
        ax.text(
            min(1.01, value + 0.012),
            index,
            f"F1={_format_decimal(value)} P={_format_decimal(precision)} R={_format_decimal(recall)}",
            va="center",
            fontsize=8,
        )
    ax.set_title(f"Top {_stage_title(prefix)} Operating Points by F1")
    ax.set_xlabel("F1")
    ax.set_xlim(0, 1.12)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_stage_comparison(best_by_stage: dict[str, dict[str, Any] | None], output_path: Path) -> None:
    rows = [(prefix, title, row) for prefix, title in STAGES if (row := best_by_stage.get(prefix)) is not None]
    if not rows:
        _plot_empty(output_path, "No Stage Metrics")
        return
    labels = [title for _prefix, title, _row in rows]
    metrics = ("precision", "recall", "F1", "balanced_accuracy")
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for offset, metric_name in enumerate(metrics):
        values = [(_metric(row, prefix, metric_name) or 0.0) * 100.0 for prefix, _title, row in rows]
        ax.bar(x + (offset - 1.5) * width, values, width=width, label=_display_metric(metric_name))
    ax.set_title("Best Supervised CADICA Metrics by Stage")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_frame_heatmap(rows: list[dict[str, Any]], output_path: Path) -> None:
    x_values = sorted({_metric_value(row.get("frame_min_degree")) for row in rows if _metric_value(row.get("frame_min_degree")) is not None})
    y_values = sorted({_metric_value(row.get("box_margin_px")) for row in rows if _metric_value(row.get("box_margin_px")) is not None})
    if not x_values or not y_values:
        _plot_empty(output_path, "No Frame Threshold Data")
        return
    matrix = np.full((len(y_values), len(x_values)), np.nan)
    for y_index, margin in enumerate(y_values):
        for x_index, threshold in enumerate(x_values):
            values = [
                _metric(row, "frame", "F1")
                for row in rows
                if _metric_value(row.get("frame_min_degree")) == threshold and _metric_value(row.get("box_margin_px")) == margin
            ]
            valid_values = [value for value in values if value is not None]
            if valid_values:
                matrix[y_index, x_index] = max(valid_values)
    _plot_heatmap(
        matrix,
        x_labels=[_format_number(value) for value in x_values],
        y_labels=[_format_number(value) for value in y_values],
        title="Max Frame F1 by Threshold and Box Margin",
        xlabel="frame_min_degree",
        ylabel="box_margin_px",
        output_path=output_path,
    )


def plot_multiview_threshold_curve(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not any(_metric(row, "multiview_patient", "F1") is not None for row in rows):
        _plot_empty(output_path, "No Multi-View Threshold Data")
        return
    thresholds = sorted({_metric_value(row.get("multiview_min_score")) for row in rows if _metric_value(row.get("multiview_min_score")) is not None})
    values: list[float] = []
    for threshold in thresholds:
        candidates = [
            _metric(row, "multiview_patient", "F1")
            for row in rows
            if _metric_value(row.get("multiview_min_score")) == threshold
        ]
        valid = [value for value in candidates if value is not None]
        values.append(max(valid) if valid else 0.0)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(thresholds, values, marker="o", color="#4c78a8")
    ax.set_title("Best Multi-View Patient F1 by Score Threshold")
    ax.set_xlabel("multiview_min_score")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_markdown_report(
    path: Path,
    *,
    summary: dict[str, Any],
    best_by_stage: dict[str, dict[str, Any] | None],
    top_primary: list[dict[str, Any]],
    pareto_primary: list[dict[str, Any]],
    primary_prefix: str,
    table_paths: dict[str, Path],
    plot_paths: dict[str, Path],
    output_root: Path,
) -> None:
    lines = [
        "# Final CADICA Benchmark Report",
        "",
        "This report summarizes supervised CADICA benchmark outputs. It uses CADICA manifest-derived labels and saved prediction JSONs only. It does not use weak labels, does not read `views.json`, and does not rerun the pipeline.",
        "",
        "## 1. Overview",
        "",
        _markdown_table([summary["overview"]], ("benchmark_root", "sweep_rows", "multiview_evaluated", "top_k")),
        "",
        "## 2. Best Operating Points",
        "",
        _markdown_table(_stage_table_rows(best_by_stage), _stage_table_columns()),
        "",
        f"![Stage comparison]({_relative_path(plot_paths['stage_comparison'], output_root)})",
        "",
        f"## 3. Top {_stage_title(primary_prefix)} Operating Points",
        "",
        _markdown_table([_compact_row(row, primary_prefix) for row in top_primary], _compact_columns()),
        "",
        f"Full table: [{_relative_path(table_paths['top_primary_operating_points'], output_root)}]({_relative_path(table_paths['top_primary_operating_points'], output_root)})",
        "",
        f"![Top F1]({_relative_path(plot_paths['top_primary_f1'], output_root)})",
        "",
        "## 4. Precision/Recall Tradeoff",
        "",
        _markdown_table([_compact_row(row, primary_prefix) for row in pareto_primary], _compact_columns()),
        "",
        f"![Precision/recall scatter]({_relative_path(plot_paths['precision_recall'], output_root)})",
        "",
        "## 5. Threshold Sensitivity",
        "",
        f"![Frame threshold heatmap]({_relative_path(plot_paths['frame_heatmap'], output_root)})",
        "",
        f"![Multi-view threshold curve]({_relative_path(plot_paths['multiview_threshold_curve'], output_root)})",
        "",
        "## 6. Files Generated",
        "",
        *_generated_files_markdown(table_paths, plot_paths, output_root),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html_report(
    path: Path,
    *,
    summary: dict[str, Any],
    best_by_stage: dict[str, dict[str, Any] | None],
    top_primary: list[dict[str, Any]],
    pareto_primary: list[dict[str, Any]],
    primary_prefix: str,
    table_paths: dict[str, Path],
    plot_paths: dict[str, Path],
    output_root: Path,
) -> None:
    body = "\n".join(
        [
            "<h1>Final CADICA Benchmark Report</h1>",
            "<p>This report summarizes supervised CADICA benchmark outputs. It uses CADICA manifest-derived labels and saved prediction JSONs only. It does not use weak labels, does not read <code>views.json</code>, and does not rerun the pipeline.</p>",
            "<h2>1. Overview</h2>",
            _html_table([summary["overview"]], ("benchmark_root", "sweep_rows", "multiview_evaluated", "top_k")),
            "<h2>2. Best Operating Points</h2>",
            _html_table(_stage_table_rows(best_by_stage), _stage_table_columns()),
            _html_image(plot_paths["stage_comparison"], output_root, "Stage comparison"),
            f"<h2>3. Top {html.escape(_stage_title(primary_prefix))} Operating Points</h2>",
            _html_table([_compact_row(row, primary_prefix) for row in top_primary], _compact_columns()),
            _html_link(table_paths["top_primary_operating_points"], output_root, "Full table"),
            _html_image(plot_paths["top_primary_f1"], output_root, "Top F1"),
            "<h2>4. Precision/Recall Tradeoff</h2>",
            _html_table([_compact_row(row, primary_prefix) for row in pareto_primary], _compact_columns()),
            _html_image(plot_paths["precision_recall"], output_root, "Precision/recall scatter"),
            "<h2>5. Threshold Sensitivity</h2>",
            _html_image(plot_paths["frame_heatmap"], output_root, "Frame threshold heatmap"),
            _html_image(plot_paths["multiview_threshold_curve"], output_root, "Multi-view threshold curve"),
            "<h2>6. Files Generated</h2>",
            _html_file_list(table_paths, plot_paths, output_root),
        ]
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Final CADICA Benchmark Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.45; margin: 32px auto; max-width: 1180px; color: #222; }}
    h1, h2 {{ color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 7px 9px; text-align: left; }}
    th {{ background: #f2f5f7; }}
    img {{ display: block; max-width: 100%; margin: 14px 0 28px; }}
    code {{ background: #f2f5f7; padding: 1px 4px; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def build_summary_payload(
    *,
    benchmark_root: Path,
    output_root: Path,
    top_k: int,
    sweep_rows: list[dict[str, Any]],
    sweep_summary: dict[str, Any],
    cadica_summary: dict[str, Any],
    best_by_stage: dict[str, dict[str, Any] | None],
    primary_prefix: str,
    best_primary: dict[str, Any],
    table_paths: dict[str, Path],
    plot_paths: dict[str, Path],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "title": "Final CADICA Benchmark Report",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overview": {
            "benchmark_root": str(benchmark_root),
            "sweep_rows": len(sweep_rows),
            "multiview_evaluated": _multiview_evaluated(sweep_rows, sweep_summary, cadica_summary),
            "top_k": top_k,
        },
        "primary_metric_level": primary_prefix,
        "best_primary_operating_point": best_primary,
        "best_by_stage": {
            prefix: None if row is None else _compact_row(row, prefix)
            for prefix, _title in STAGES
            for row in [best_by_stage.get(prefix)]
        },
        "source_summaries": {
            "cadica_threshold_sweep_summary": sweep_summary,
            "cadica_summary": cadica_summary,
        },
        "warnings": warnings,
        "tables": {name: str(path) for name, path in table_paths.items()},
        "plots": {name: str(path) for name, path in plot_paths.items()},
    }


def _build_warnings(rows: list[dict[str, Any]], sweep_summary: dict[str, Any], benchmark_root: Path) -> list[str]:
    warnings: list[str] = []
    if not _multiview_evaluated(rows, sweep_summary, {}):
        warnings.append("Multi-view metrics were not found; the report uses patient aggregation as the primary level.")
    if not (benchmark_root / MULTIVIEW_PATIENT_ROWS_CSV).is_file() and _multiview_evaluated(rows, sweep_summary, {}):
        warnings.append("Multi-view patient row CSV is missing; selected case review table was not generated.")
    if all((_metric(row, "multiview_patient", "recall") or 0.0) == 0.0 for row in rows if _metric(row, "multiview_patient", "recall") is not None):
        if _multiview_evaluated(rows, sweep_summary, {}):
            warnings.append("All multi-view patient recall values are zero; check the selected threshold range and saved multi-view outputs.")
    return warnings


def _best_by_stage(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    return {prefix: (_rank_rows(rows, prefix)[0] if _rank_rows(rows, prefix) else None) for prefix, _title in STAGES}


def _rank_rows(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    candidates = [row for row in rows if _metric(row, prefix, "F1") is not None]
    return sorted(
        candidates,
        key=lambda row: (
            -_ranking_value(_metric(row, prefix, "F1")),
            -_ranking_value(_metric(row, prefix, "recall")),
            -_ranking_value(_metric(row, prefix, "precision")),
            -_ranking_value(_metric(row, prefix, "balanced_accuracy")),
            -_ranking_value(_metric(row, prefix, "specificity")),
            _ranking_count(_metric(row, prefix, "FP")),
            _ranking_count(_metric(row, prefix, "FN")),
            _ranking_value(row.get("frame_min_degree")),
            _ranking_value(row.get("multiview_min_score")),
        ),
    )


def _pareto_rows(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for candidate in rows:
        precision = _metric(candidate, prefix, "precision")
        recall = _metric(candidate, prefix, "recall")
        if precision is None or recall is None:
            continue
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            other_precision = _metric(other, prefix, "precision")
            other_recall = _metric(other, prefix, "recall")
            if other_precision is None or other_recall is None:
                continue
            if other_precision >= precision and other_recall >= recall and (other_precision > precision or other_recall > recall):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return _rank_rows(front, prefix)


def _stage_table_rows(best_by_stage: dict[str, dict[str, Any] | None]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prefix, title in STAGES:
        row = best_by_stage.get(prefix)
        if row is None:
            continue
        rows.append(
            {
                "stage": title,
                "frame_min_degree": row.get("frame_min_degree"),
                "box_margin_px": row.get("box_margin_px"),
                "video_prediction_source": row.get("video_prediction_source"),
                "multiview_min_score": row.get("multiview_min_score"),
                "precision": _format_decimal(_metric(row, prefix, "precision")),
                "recall": _format_decimal(_metric(row, prefix, "recall")),
                "specificity": _format_decimal(_metric(row, prefix, "specificity")),
                "F1": _format_decimal(_metric(row, prefix, "F1")),
                "balanced_accuracy": _format_decimal(_metric(row, prefix, "balanced_accuracy")),
                "TP": _format_count(_metric(row, prefix, "TP")),
                "FP": _format_count(_metric(row, prefix, "FP")),
                "TN": _format_count(_metric(row, prefix, "TN")),
                "FN": _format_count(_metric(row, prefix, "FN")),
            }
        )
    return rows


def _stage_table_columns() -> tuple[str, ...]:
    return (
        "stage",
        "frame_min_degree",
        "box_margin_px",
        "video_prediction_source",
        "multiview_min_score",
        "precision",
        "recall",
        "specificity",
        "F1",
        "balanced_accuracy",
        "TP",
        "FP",
        "TN",
        "FN",
    )


def _compact_row(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        "config": _compact_config_label(row),
        "precision": _format_decimal(_metric(row, prefix, "precision")),
        "recall": _format_decimal(_metric(row, prefix, "recall")),
        "specificity": _format_decimal(_metric(row, prefix, "specificity")),
        "F1": _format_decimal(_metric(row, prefix, "F1")),
        "balanced_accuracy": _format_decimal(_metric(row, prefix, "balanced_accuracy")),
        "TP": _format_count(_metric(row, prefix, "TP")),
        "FP": _format_count(_metric(row, prefix, "FP")),
        "TN": _format_count(_metric(row, prefix, "TN")),
        "FN": _format_count(_metric(row, prefix, "FN")),
    }


def _compact_columns() -> tuple[str, ...]:
    return ("config", "precision", "recall", "specificity", "F1", "balanced_accuracy", "TP", "FP", "TN", "FN")


def _selected_multiview_rows(path: Path, selected_sweep_row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if selected_sweep_row is None or not path.is_file():
        return []
    threshold = _metric_value(selected_sweep_row.get("multiview_min_score")) or 0.0
    rows = _read_csv(path)
    selected_rows: list[dict[str, Any]] = []
    for row in rows:
        selected = dict(row)
        label = _bool_value(row.get("label_positive"))
        base_predicted = _bool_value(row.get("predicted_positive"))
        score = _metric_value(row.get("score")) or 0.0
        predicted = bool(base_predicted and score >= threshold)
        selected["selected_multiview_min_score"] = threshold
        selected["selected_predicted_positive"] = predicted
        selected["selected_outcome"] = "" if label is None else _outcome(label, predicted)
        selected_rows.append(selected)
    return selected_rows


def _metric(row: dict[str, Any], prefix: str, metric_name: str) -> float | None:
    key = f"{prefix}_{metric_name}"
    if metric_name == "F1":
        key = f"{prefix}_F1"
    return _metric_value(row.get(key))


def _metric_value(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value) if isfinite(float(value)) else None
    try:
        number = float(str(value))
    except ValueError:
        return None
    return number if isfinite(number) else None


def _ranking_value(value: Any) -> float:
    number = _metric_value(value)
    return float("-inf") if number is None else number


def _ranking_count(value: Any) -> int:
    number = _metric_value(value)
    return 10**12 if number is None else int(number)


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _outcome(label: bool, predicted: bool) -> str:
    if label and predicted:
        return "TP"
    if not label and predicted:
        return "FP"
    if not label and not predicted:
        return "TN"
    return "FN"


def _multiview_evaluated(rows: list[dict[str, Any]], sweep_summary: dict[str, Any], cadica_summary: dict[str, Any]) -> bool:
    summary_config = sweep_summary.get("config") if isinstance(sweep_summary.get("config"), dict) else {}
    cadica_config = cadica_summary.get("config") if isinstance(cadica_summary.get("config"), dict) else {}
    if summary_config.get("multiview_evaluated") is True or cadica_config.get("multiview_evaluated") is True:
        return True
    return any(_metric(row, "multiview_patient", "total_evaluated") is not None for row in rows)


def _compact_config_label(row: dict[str, Any]) -> str:
    parts = [
        f"fd={_format_number(row.get('frame_min_degree'))}",
        f"box={_format_number(row.get('box_margin_px'))}",
    ]
    source = str(row.get("video_prediction_source") or "")
    if source:
        parts.append(f"video={source}")
    if row.get("multiview_min_score") not in (None, ""):
        parts.append(f"mv={_format_number(row.get('multiview_min_score'))}")
    return ", ".join(parts)


def _stage_title(prefix: str) -> str:
    for stage_prefix, title in STAGES:
        if stage_prefix == prefix:
            return title
    return prefix


def _display_metric(metric_name: str) -> str:
    if metric_name == "F1":
        return "F1"
    return metric_name.replace("_", " ").title()


def _format_decimal(value: Any) -> str:
    number = _metric_value(value)
    return "n/a" if number is None else f"{number:.3f}"


def _format_count(value: Any) -> str:
    number = _metric_value(value)
    return "n/a" if number is None else str(int(number))


def _format_number(value: Any) -> str:
    number = _metric_value(value)
    if number is None:
        return "n/a"
    if float(number).is_integer():
        return str(int(number))
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _plot_heatmap(
    matrix: np.ndarray,
    *,
    x_labels: list[str],
    y_labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if matrix.size == 0 or np.all(np.isnan(matrix)):
        _plot_empty(output_path, title)
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for y_index in range(matrix.shape[0]):
        for x_index in range(matrix.shape[1]):
            value = matrix[y_index, x_index]
            if not np.isnan(value):
                ax.text(x_index, y_index, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="F1")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_empty(output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _markdown_table(rows: list[dict[str, Any]], columns: Iterable[str]) -> str:
    column_list = list(columns)
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(column_list) + " |",
        "| " + " | ".join("---" for _ in column_list) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in column_list) + " |")
    return "\n".join(lines)


def _html_table(rows: list[dict[str, Any]], columns: Iterable[str]) -> str:
    column_list = list(columns)
    if not rows:
        return "<p><em>No rows.</em></p>"
    header = "".join(f"<th>{html.escape(column)}</th>" for column in column_list)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in column_list) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _html_image(path: Path, output_root: Path, alt: str) -> str:
    return f'<img src="{html.escape(_relative_path(path, output_root))}" alt="{html.escape(alt)}">'


def _html_link(path: Path, output_root: Path, label: str) -> str:
    return f'<a href="{html.escape(_relative_path(path, output_root))}">{html.escape(label)}</a>'


def _html_file_list(table_paths: dict[str, Path], plot_paths: dict[str, Path], output_root: Path) -> str:
    items = "".join(
        f"<li>{_html_link(path, output_root, _relative_path(path, output_root))}</li>"
        for path in (*table_paths.values(), *plot_paths.values())
    )
    return f"<ul>{items}</ul>"


def _generated_files_markdown(table_paths: dict[str, Path], plot_paths: dict[str, Path], output_root: Path) -> list[str]:
    return [
        f"- [{_relative_path(path, output_root)}]({_relative_path(path, output_root)})"
        for path in (*table_paths.values(), *plot_paths.values())
    ]


def _relative_path(path: Path, output_root: Path) -> str:
    try:
        return path.relative_to(output_root).as_posix()
    except ValueError:
        return path.as_posix()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fieldnames})


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
