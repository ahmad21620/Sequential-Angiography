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


LEVELS = ("frame", "temporal", "multiview")
SUMMARY_FILENAMES = {
    "frame": "frame_summary.json",
    "temporal": "temporal_sequence_summary.json",
    "multiview": "multiview_case_summary.json",
}
ROW_FILENAMES = {
    "frame": "frame_rows.csv",
    "temporal": "temporal_sequence_rows.csv",
    "multiview": "multiview_case_rows.csv",
}
FRAME_PARAMS = (
    "radius_outside_fraction_threshold",
    "radius_min_outside_samples",
    "stenosis_threshold",
    "average_radius_threshold",
)
TEMPORAL_PARAMS = (
    "min_supporting_frames",
    "min_persistence_ratio",
)
BINARY_METRICS = (
    "total_evaluated",
    "positive_labels",
    "negative_labels",
    "predicted_positive",
    "predicted_negative",
    "TP",
    "FP",
    "TN",
    "FN",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "balanced_accuracy",
    "false_positive_rate",
    "false_negative_rate",
    "negative_predictive_value",
    "skipped_missing_label",
    "skipped_unclear_label",
    "labels_without_prediction",
)
ALL_VARIANT_FIELDS = (
    "level",
    "frame_variant",
    "temporal_variant",
    "result_summary_path",
    "row_count",
    *FRAME_PARAMS,
    *TEMPORAL_PARAMS,
    *BINARY_METRICS,
)
SENSITIVITY_FIELDS = (
    "parameter",
    "value",
    "count",
    "mean_f1",
    "max_f1",
    "mean_precision",
    "max_precision",
    "mean_recall",
    "max_recall",
)
FINAL_CASE_FIELDS = (
    "case_id",
    "predicted_positive",
    "label_target",
    "weak_outcome",
    "confidence_score",
    "final_degree",
    "final_max_degree",
    "supporting_view_count",
    "distinct_supporting_view_count",
    "source_path",
)


class SweepReportError(RuntimeError):
    """Raised when a final sweep report cannot be generated."""


@dataclass(frozen=True, slots=True)
class VariantMetric:
    level: str
    frame_variant: str | None
    temporal_variant: str | None
    result_summary_path: Path
    row_count: int | None
    frame_params: dict[str, int | float | None]
    temporal_params: dict[str, int | float | None]
    metrics: dict[str, int | float | None]
    rows_csv_path: Path | None = None

    def to_table_row(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "frame_variant": self.frame_variant,
            "temporal_variant": self.temporal_variant,
            "result_summary_path": str(self.result_summary_path),
            "row_count": self.row_count,
            **self.frame_params,
            **self.temporal_params,
            **self.metrics,
        }

    def metric(self, name: str) -> int | float | None:
        return self.metrics.get(name)


@dataclass(frozen=True, slots=True)
class ReportArtifacts:
    output_root: Path
    tables: dict[str, Path]
    plots: dict[str, Path]
    reports: dict[str, Path]
    warnings: list[str]
    best_multiview: VariantMetric


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a compact final binary stenosis detection report from existing sweep benchmark outputs.",
    )
    parser.add_argument("--sweep-root", required=True, help="Sweep root used to run the original parameter sweep.")
    parser.add_argument(
        "--benchmark-root",
        help="Benchmark output root. Defaults to <sweep-root>/benchmark_results.",
    )
    parser.add_argument(
        "--output-root",
        help="Final report output root. Defaults to <benchmark-root>/final_report.",
    )
    parser.add_argument(
        "--primary-level",
        default="multiview",
        choices=("multiview",),
        help="Primary final benchmark level. Only multiview is supported for the final report.",
    )
    parser.add_argument("--top-k", type=int, default=25, help="Number of top multi-view variants to show.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        artifacts = run_sweep_report(
            sweep_root=args.sweep_root,
            benchmark_root=args.benchmark_root,
            output_root=args.output_root,
            primary_level=args.primary_level,
            top_k=args.top_k,
        )
    except (SweepReportError, FileNotFoundError, NotADirectoryError, ValueError, OSError) as exc:
        print(f"Sweep report failed: {exc}", file=sys.stderr)
        return 2

    for warning in artifacts.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    print(f"Final sweep benchmark report: {artifacts.reports['markdown']}")
    print(f"HTML report: {artifacts.reports['html']}")
    print(f"Summary JSON: {artifacts.reports['summary_json']}")
    return 0


def run_sweep_report(
    *,
    sweep_root: str | Path,
    benchmark_root: str | Path | None = None,
    output_root: str | Path | None = None,
    primary_level: str = "multiview",
    top_k: int = 25,
) -> ReportArtifacts:
    if primary_level != "multiview":
        raise ValueError("Final sweep reports currently support only --primary-level multiview.")
    if top_k <= 0:
        raise ValueError("--top-k must be positive.")

    resolved_sweep_root = Path(sweep_root)
    if not resolved_sweep_root.exists():
        raise FileNotFoundError(f"Sweep root does not exist: {resolved_sweep_root}")
    if not resolved_sweep_root.is_dir():
        raise NotADirectoryError(f"Sweep root is not a directory: {resolved_sweep_root}")

    resolved_benchmark_root = Path(benchmark_root) if benchmark_root is not None else resolved_sweep_root / "benchmark_results"
    if not resolved_benchmark_root.exists():
        raise FileNotFoundError(f"Benchmark root does not exist: {resolved_benchmark_root}")
    if not resolved_benchmark_root.is_dir():
        raise NotADirectoryError(f"Benchmark root is not a directory: {resolved_benchmark_root}")

    resolved_output_root = Path(output_root) if output_root is not None else resolved_benchmark_root / "final_report"
    tables_root = resolved_output_root / "tables"
    plots_root = resolved_output_root / "plots"
    tables_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    variants, benchmark_overview = load_variant_metrics(resolved_benchmark_root)
    multiview_variants = [variant for variant in variants if variant.level == "multiview"]
    if not multiview_variants:
        raise SweepReportError(f"No multi-view benchmark summaries found under: {resolved_benchmark_root}")

    ranked_multiview = rank_variants(multiview_variants)
    best_multiview = ranked_multiview[0]
    top_multiview = ranked_multiview[:top_k]
    best_by_level = _best_by_level(variants)
    pareto_multiview = pareto_front(multiview_variants)
    sensitivity_rows = metric_sensitivity(multiview_variants)
    warnings = build_warnings(multiview_variants, best_multiview)

    table_paths = {
        "all_variant_metrics": tables_root / "all_variant_metrics.csv",
        "top_multiview_variants": tables_root / "top_multiview_variants.csv",
        "best_by_level": tables_root / "best_by_level.csv",
        "pareto_multiview_variants": tables_root / "pareto_multiview_variants.csv",
        "metric_sensitivity": tables_root / "metric_sensitivity.csv",
        "final_selected_variant_cases": tables_root / "final_selected_variant_cases.csv",
    }
    _write_csv(table_paths["all_variant_metrics"], [variant.to_table_row() for variant in variants], ALL_VARIANT_FIELDS)
    _write_csv(table_paths["top_multiview_variants"], [variant.to_table_row() for variant in top_multiview], ALL_VARIANT_FIELDS)
    _write_csv(table_paths["best_by_level"], [variant.to_table_row() for variant in best_by_level], ALL_VARIANT_FIELDS)
    _write_csv(
        table_paths["pareto_multiview_variants"],
        [variant.to_table_row() for variant in pareto_multiview],
        ALL_VARIANT_FIELDS,
    )
    _write_csv(table_paths["metric_sensitivity"], sensitivity_rows, SENSITIVITY_FIELDS)

    if not write_final_selected_variant_cases(best_multiview, table_paths["final_selected_variant_cases"]):
        warnings.append(
            "Benchmark case rows are missing for the selected best multi-view variant; case review CSV is header-only."
        )

    plot_paths = {
        "top_multiview_f1": plots_root / "01_top_multiview_f1.png",
        "precision_recall_scatter": plots_root / "02_multiview_precision_recall_scatter.png",
        "confusion_matrix": plots_root / "03_best_multiview_confusion_matrix.png",
        "best_by_level_comparison": plots_root / "04_best_by_level_comparison.png",
        "temporal_parameter_heatmap": plots_root / "05_temporal_parameter_heatmap.png",
        "frame_threshold_heatmap": plots_root / "06_frame_threshold_heatmap.png",
        "radius_parameter_heatmap": plots_root / "07_radius_parameter_heatmap.png",
    }
    plot_top_multiview_f1(top_multiview, plot_paths["top_multiview_f1"])
    plot_precision_recall_scatter(
        multiview_variants,
        pareto_multiview,
        best_multiview,
        plot_paths["precision_recall_scatter"],
    )
    plot_confusion_matrix(best_multiview, plot_paths["confusion_matrix"])
    plot_best_by_level(best_by_level, plot_paths["best_by_level_comparison"])
    plot_heatmap(
        multiview_variants,
        x_param="min_persistence_ratio",
        y_param="min_supporting_frames",
        output_path=plot_paths["temporal_parameter_heatmap"],
        title="Multi-View Max F1 By Temporal Fusion Parameters",
        xlabel="Min Persistence Ratio",
        ylabel="Min Supporting Frames",
    )
    plot_heatmap(
        multiview_variants,
        x_param="average_radius_threshold",
        y_param="stenosis_threshold",
        output_path=plot_paths["frame_threshold_heatmap"],
        title="Multi-View Max F1 By Stenosis And Radius Thresholds",
        xlabel="Average Radius Threshold",
        ylabel="Stenosis Threshold",
    )
    plot_heatmap(
        multiview_variants,
        x_param="radius_min_outside_samples",
        y_param="radius_outside_fraction_threshold",
        output_path=plot_paths["radius_parameter_heatmap"],
        title="Multi-View Max F1 By Radius Outside Parameters",
        xlabel="Radius Min Outside Samples",
        ylabel="Radius Outside Fraction Threshold",
    )

    summary_payload = build_summary_payload(
        sweep_root=resolved_sweep_root,
        benchmark_root=resolved_benchmark_root,
        output_root=resolved_output_root,
        top_k=top_k,
        variants=variants,
        best_multiview=best_multiview,
        best_by_level=best_by_level,
        benchmark_overview=benchmark_overview,
        table_paths=table_paths,
        plot_paths=plot_paths,
        warnings=warnings,
    )
    report_paths = {
        "markdown": resolved_output_root / "report.md",
        "html": resolved_output_root / "report.html",
        "summary_json": resolved_output_root / "summary.json",
    }
    write_markdown_report(
        report_paths["markdown"],
        variants=variants,
        best_multiview=best_multiview,
        top_multiview=top_multiview[:10],
        pareto_multiview=pareto_multiview,
        best_by_level=best_by_level,
        sensitivity_rows=sensitivity_rows,
        benchmark_overview=benchmark_overview,
        table_paths=table_paths,
        plot_paths=plot_paths,
        output_root=resolved_output_root,
    )
    write_html_report(
        report_paths["html"],
        variants=variants,
        best_multiview=best_multiview,
        top_multiview=top_multiview[:10],
        pareto_multiview=pareto_multiview,
        best_by_level=best_by_level,
        sensitivity_rows=sensitivity_rows,
        benchmark_overview=benchmark_overview,
        table_paths=table_paths,
        plot_paths=plot_paths,
        output_root=resolved_output_root,
    )
    report_paths["summary_json"].write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    return ReportArtifacts(
        output_root=resolved_output_root,
        tables=table_paths,
        plots=plot_paths,
        reports=report_paths,
        warnings=warnings,
        best_multiview=best_multiview,
    )


def load_variant_metrics(benchmark_root: str | Path) -> tuple[list[VariantMetric], dict[str, Any]]:
    resolved_root = Path(benchmark_root)
    summary_path = resolved_root / "sweep_benchmark_summary.json"
    overview: dict[str, Any] = {}
    variants_by_path: dict[Path, VariantMetric] = {}

    if summary_path.is_file():
        payload = _read_json(summary_path)
        overview = {
            "summary_path": str(summary_path),
            "total_jobs": _coerce_int(payload.get("total_jobs")),
            "completed_jobs": _coerce_int(payload.get("completed_jobs")),
            "skipped_jobs": _coerce_int(payload.get("skipped_jobs")),
            "failed_jobs": _coerce_int(payload.get("failed_jobs")),
        }
        for entry in _summary_entries(payload):
            variant = _variant_from_summary_entry(entry, resolved_root)
            if variant is not None:
                variants_by_path[variant.result_summary_path.resolve()] = variant

    for discovered in _discover_summary_paths(resolved_root):
        key = discovered.resolve()
        if key not in variants_by_path:
            variants_by_path[key] = _variant_from_summary_path(discovered, resolved_root)

    variants = sorted(
        variants_by_path.values(),
        key=lambda variant: (
            LEVELS.index(variant.level),
            variant.frame_variant or "",
            variant.temporal_variant or "",
            str(variant.result_summary_path),
        ),
    )
    if not overview:
        overview = {
            "summary_path": None,
            "total_jobs": None,
            "completed_jobs": None,
            "skipped_jobs": None,
            "failed_jobs": None,
            "discovered_summary_count": len(variants),
        }
    return variants, overview


def parse_frame_variant(frame_variant: str | None) -> dict[str, int | float | None]:
    values = {name: None for name in FRAME_PARAMS}
    if not frame_variant:
        return values
    parsed = _parse_variant_parts(frame_variant, FRAME_PARAMS)
    values.update(
        {
            "radius_outside_fraction_threshold": _as_float(parsed.get("radius_outside_fraction_threshold")),
            "radius_min_outside_samples": _as_int(parsed.get("radius_min_outside_samples")),
            "stenosis_threshold": _as_float(parsed.get("stenosis_threshold")),
            "average_radius_threshold": _as_float(parsed.get("average_radius_threshold")),
        }
    )
    return values


def parse_temporal_variant(temporal_variant: str | None) -> dict[str, int | float | None]:
    values = {name: None for name in TEMPORAL_PARAMS}
    if not temporal_variant:
        return values
    parsed = _parse_variant_parts(temporal_variant, TEMPORAL_PARAMS)
    values.update(
        {
            "min_supporting_frames": _as_int(parsed.get("min_supporting_frames")),
            "min_persistence_ratio": _as_float(parsed.get("min_persistence_ratio")),
        }
    )
    return values


def rank_variants(variants: Iterable[VariantMetric]) -> list[VariantMetric]:
    return sorted(
        variants,
        key=lambda variant: (
            -_ranking_float(variant.metric("f1")),
            -_ranking_float(variant.metric("recall")),
            -_ranking_float(variant.metric("precision")),
            -_ranking_float(variant.metric("balanced_accuracy")),
            _ranking_count(variant.metric("FP")),
            _ranking_count(variant.metric("FN")),
            variant.frame_variant or "",
            variant.temporal_variant or "",
        ),
    )


def pareto_front(variants: Iterable[VariantMetric]) -> list[VariantMetric]:
    candidates = [
        variant
        for variant in variants
        if _is_number(variant.metric("precision")) and _is_number(variant.metric("recall"))
    ]
    front: list[VariantMetric] = []
    for candidate in candidates:
        precision = float(candidate.metric("precision"))
        recall = float(candidate.metric("recall"))
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            other_precision = float(other.metric("precision"))
            other_recall = float(other.metric("recall"))
            if (
                other_precision >= precision
                and other_recall >= recall
                and (other_precision > precision or other_recall > recall)
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return rank_variants(front)


def metric_sensitivity(variants: Iterable[VariantMetric]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variant_list = list(variants)
    for parameter in (*FRAME_PARAMS, *TEMPORAL_PARAMS):
        values = sorted(
            {
                variant.to_table_row().get(parameter)
                for variant in variant_list
                if variant.to_table_row().get(parameter) is not None
            },
            key=lambda value: float(value),
        )
        for value in values:
            group = [variant for variant in variant_list if variant.to_table_row().get(parameter) == value]
            rows.append(
                {
                    "parameter": parameter,
                    "value": value,
                    "count": len(group),
                    "mean_f1": _mean_metric(group, "f1"),
                    "max_f1": _max_metric(group, "f1"),
                    "mean_precision": _mean_metric(group, "precision"),
                    "max_precision": _max_metric(group, "precision"),
                    "mean_recall": _mean_metric(group, "recall"),
                    "max_recall": _max_metric(group, "recall"),
                }
            )
    return rows


def build_warnings(multiview_variants: list[VariantMetric], best_multiview: VariantMetric) -> list[str]:
    warnings: list[str] = []
    if all(_coerce_int(variant.metric("predicted_positive")) == 0 for variant in multiview_variants):
        warnings.append(
            "All multi-view variants have predicted_positive = 0. This can indicate an overly strict sweep or a split-side benchmark parsing issue."
        )
    if all(_safe_float(variant.metric("recall")) == 0.0 for variant in multiview_variants):
        warnings.append(
            "All multi-view variants have recall = 0. If split-by-coronary-side fusion was used, confirm the benchmark was rerun with split-output support."
        )
    split_rows_detected = _best_rows_have_split_output(best_multiview)
    if split_rows_detected and _safe_float(best_multiview.metric("recall")) == 0.0:
        warnings.append(
            "Split-by-coronary-side benchmark rows were detected and the selected variant has recall = 0."
        )

    total_predictions = sum(
        (_coerce_int(variant.metric("total_evaluated")) or 0)
        + (_coerce_int(variant.metric("skipped_missing_label")) or 0)
        + (_coerce_int(variant.metric("skipped_unclear_label")) or 0)
        for variant in multiview_variants
    )
    skipped_labels = sum(
        (_coerce_int(variant.metric("skipped_missing_label")) or 0)
        + (_coerce_int(variant.metric("skipped_unclear_label")) or 0)
        for variant in multiview_variants
    )
    if total_predictions > 0 and skipped_labels / total_predictions >= 0.10:
        warnings.append(
            f"Many multi-view rows have missing or unclear labels: {skipped_labels} skipped across {total_predictions} evaluated rows."
        )
    return warnings


def write_final_selected_variant_cases(variant: VariantMetric, output_path: Path) -> bool:
    source_path = variant.rows_csv_path
    source_has_selected_side = False
    if source_path is None or not source_path.is_file():
        _write_csv(output_path, [], FINAL_CASE_FIELDS)
        return False

    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        source_has_selected_side = "selected_side" in (reader.fieldnames or [])
        fieldnames = (*FINAL_CASE_FIELDS[:-1], "selected_side", FINAL_CASE_FIELDS[-1]) if source_has_selected_side else FINAL_CASE_FIELDS
        rows = [{field: row.get(field) for field in fieldnames} for row in reader]
    _write_csv(output_path, rows, fieldnames)
    return True


def plot_top_multiview_f1(variants: list[VariantMetric], output_path: Path) -> None:
    labels = [_compact_variant_label(variant) for variant in variants][::-1]
    f1_values = [_safe_float(variant.metric("f1")) for variant in variants][::-1]
    precision_values = [_safe_float(variant.metric("precision")) for variant in variants][::-1]
    recall_values = [_safe_float(variant.metric("recall")) for variant in variants][::-1]
    height = max(5.0, 0.34 * max(1, len(variants)) + 1.2)
    fig, ax = plt.subplots(figsize=(12, height))
    y_positions = np.arange(len(labels))
    ax.barh(y_positions, f1_values, color="#2a9d8f", alpha=0.88)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, max(1.0, max(f1_values, default=0.0) + 0.08))
    ax.set_xlabel("F1")
    ax.set_title(f"Top {len(variants)} Multi-View Variants By Binary F1")
    for index, (f1_value, precision, recall) in enumerate(zip(f1_values, precision_values, recall_values)):
        ax.text(
            f1_value + 0.01,
            index,
            f"F1 {f1_value:.3f}  P {precision:.3f}  R {recall:.3f}",
            va="center",
            fontsize=8,
        )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_precision_recall_scatter(
    variants: list[VariantMetric],
    pareto_variants: list[VariantMetric],
    best_variant: VariantMetric,
    output_path: Path,
) -> None:
    valid = [variant for variant in variants if _is_number(variant.metric("precision")) and _is_number(variant.metric("recall"))]
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    if valid:
        recalls = [float(variant.metric("recall")) for variant in valid]
        precisions = [float(variant.metric("precision")) for variant in valid]
        f1_values = [_safe_float(variant.metric("f1")) for variant in valid]
        scatter = ax.scatter(recalls, precisions, c=f1_values, cmap="viridis", s=26, alpha=0.60, edgecolors="none")
        fig.colorbar(scatter, ax=ax, label="F1")

    pareto_set = {variant.result_summary_path for variant in pareto_variants}
    pareto_valid = [variant for variant in valid if variant.result_summary_path in pareto_set]
    if pareto_valid:
        ax.scatter(
            [float(variant.metric("recall")) for variant in pareto_valid],
            [float(variant.metric("precision")) for variant in pareto_valid],
            facecolors="none",
            edgecolors="#f4a261",
            s=84,
            linewidths=1.8,
            label="Pareto front",
        )
    if _is_number(best_variant.metric("precision")) and _is_number(best_variant.metric("recall")):
        ax.scatter(
            [float(best_variant.metric("recall"))],
            [float(best_variant.metric("precision"))],
            marker="*",
            s=220,
            color="#e76f51",
            edgecolors="black",
            linewidths=0.5,
            label="Top F1",
            zorder=5,
        )

    for variant in rank_variants(valid)[:5]:
        ax.annotate(
            _compact_variant_label(variant),
            (float(variant.metric("recall")), float(variant.metric("precision"))),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.75, "ec": "none"},
        )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Multi-View Binary Precision/Recall Tradeoff")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_confusion_matrix(variant: VariantMetric, output_path: Path) -> None:
    tp = _coerce_int(variant.metric("TP")) or 0
    fp = _coerce_int(variant.metric("FP")) or 0
    tn = _coerce_int(variant.metric("TN")) or 0
    fn = _coerce_int(variant.metric("FN")) or 0
    matrix = np.array([[tp, fn], [fp, tn]], dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    image = ax.imshow(matrix, cmap="YlGnBu")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Label +", "Label -"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Pred +", "Pred -"])
    labels = np.array([["TP", "FN"], ["FP", "TN"]])
    for row_index in range(2):
        for col_index in range(2):
            ax.text(
                col_index,
                row_index,
                f"{labels[row_index, col_index]}\n{int(matrix[row_index, col_index])}",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="black",
            )
    ax.set_title(
        "Best Multi-View Confusion Matrix\n"
        f"Precision {_format_metric(variant.metric('precision'))} | "
        f"Recall {_format_metric(variant.metric('recall'))} | "
        f"F1 {_format_metric(variant.metric('f1'))}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_best_by_level(variants: list[VariantMetric], output_path: Path) -> None:
    if not variants:
        _plot_empty(output_path, "No Stage Comparison Data")
        return
    levels = [variant.level.title() for variant in variants]
    metrics = ("precision", "recall", "f1")
    x_positions = np.arange(len(levels))
    width = 0.23
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = ("#2a9d8f", "#e9c46a", "#e76f51")
    for index, metric in enumerate(metrics):
        values = [_safe_float(variant.metric(metric)) for variant in variants]
        ax.bar(x_positions + (index - 1) * width, values, width, label=metric.title(), color=colors[index])
    ax.set_xticks(x_positions)
    ax.set_xticklabels(levels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric Value")
    ax.set_title("Best Variant By Stage (Final Result Is Multi-View)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_heatmap(
    variants: list[VariantMetric],
    *,
    x_param: str,
    y_param: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    rows = [variant.to_table_row() for variant in variants]
    x_values = sorted({row.get(x_param) for row in rows if row.get(x_param) is not None}, key=lambda value: float(value))
    y_values = sorted({row.get(y_param) for row in rows if row.get(y_param) is not None}, key=lambda value: float(value))
    if not x_values or not y_values:
        _plot_empty(output_path, title)
        return

    grid = np.full((len(y_values), len(x_values)), np.nan)
    for y_index, y_value in enumerate(y_values):
        for x_index, x_value in enumerate(x_values):
            group = [
                variant
                for variant in variants
                if variant.to_table_row().get(x_param) == x_value and variant.to_table_row().get(y_param) == y_value
            ]
            value = _max_metric(group, "f1")
            if value is not None:
                grid[y_index, x_index] = value

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    image = ax.imshow(grid, cmap="viridis", vmin=0.0, vmax=max(1.0, np.nanmax(grid) if not np.isnan(grid).all() else 1.0))
    fig.colorbar(image, ax=ax, label="Max F1")
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([_format_param_value(value) for value in x_values])
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels([_format_param_value(value) for value in y_values])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for y_index in range(len(y_values)):
        for x_index in range(len(x_values)):
            if not np.isnan(grid[y_index, x_index]):
                ax.text(x_index, y_index, f"{grid[y_index, x_index]:.2f}", ha="center", va="center", color="white", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=230)
    plt.close(fig)


def write_markdown_report(
    path: Path,
    *,
    variants: list[VariantMetric],
    best_multiview: VariantMetric,
    top_multiview: list[VariantMetric],
    pareto_multiview: list[VariantMetric],
    best_by_level: list[VariantMetric],
    sensitivity_rows: list[dict[str, Any]],
    benchmark_overview: dict[str, Any],
    table_paths: dict[str, Path],
    plot_paths: dict[str, Path],
    output_root: Path,
) -> None:
    variant_counts = _variant_counts_by_level(variants)
    lines = [
        "# Final CADICA Sweep Benchmark Report",
        "",
        "## 1. Sweep overview",
        "",
        "This report summarizes binary weak-label agreement for stenosis detection. Severity-related tables and level-of-stenosis statistics are intentionally excluded.",
        "",
        _markdown_table(
            [{"level": level, "benchmarked_variants": variant_counts.get(level, 0)} for level in LEVELS],
            ("level", "benchmarked_variants"),
        ),
        "",
        _job_summary_sentence(benchmark_overview),
        "",
        "## 2. Best final multi-view result",
        "",
        _markdown_table([_best_result_row(best_multiview)], _best_result_columns()),
        "",
        f"![Best multi-view confusion matrix]({_relative_path(plot_paths['confusion_matrix'], output_root)})",
        "",
        f"![Multi-view precision/recall scatter]({_relative_path(plot_paths['precision_recall_scatter'], output_root)})",
        "",
        "## 3. Top multi-view variants",
        "",
        _markdown_table([_compact_metric_row(variant) for variant in top_multiview], _compact_metric_columns()),
        "",
        f"Full CSV: [{_relative_path(table_paths['top_multiview_variants'], output_root)}]({_relative_path(table_paths['top_multiview_variants'], output_root)})",
        "",
        "## 4. Precision/recall tradeoff",
        "",
        _markdown_table([_compact_metric_row(variant) for variant in pareto_multiview[:10]], _compact_metric_columns()),
        "",
        f"Scatter plot: [{_relative_path(plot_paths['precision_recall_scatter'], output_root)}]({_relative_path(plot_paths['precision_recall_scatter'], output_root)})",
        "",
        "## 5. Stage comparison",
        "",
        "Multi-view is the final case-level result. Frame and temporal rows are included only as stage comparisons.",
        "",
        _markdown_table([_compact_metric_row(variant) for variant in best_by_level], _compact_metric_columns()),
        "",
        f"![Stage comparison]({_relative_path(plot_paths['best_by_level_comparison'], output_root)})",
        "",
        "## 6. Parameter sensitivity",
        "",
        _sensitivity_summary(sensitivity_rows),
        "",
        f"- Temporal heatmap: [{_relative_path(plot_paths['temporal_parameter_heatmap'], output_root)}]({_relative_path(plot_paths['temporal_parameter_heatmap'], output_root)})",
        f"- Frame threshold heatmap: [{_relative_path(plot_paths['frame_threshold_heatmap'], output_root)}]({_relative_path(plot_paths['frame_threshold_heatmap'], output_root)})",
        f"- Radius parameter heatmap: [{_relative_path(plot_paths['radius_parameter_heatmap'], output_root)}]({_relative_path(plot_paths['radius_parameter_heatmap'], output_root)})",
        "",
        "## 7. Files generated",
        "",
        *_generated_files_markdown(table_paths, plot_paths, output_root),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html_report(
    path: Path,
    *,
    variants: list[VariantMetric],
    best_multiview: VariantMetric,
    top_multiview: list[VariantMetric],
    pareto_multiview: list[VariantMetric],
    best_by_level: list[VariantMetric],
    sensitivity_rows: list[dict[str, Any]],
    benchmark_overview: dict[str, Any],
    table_paths: dict[str, Path],
    plot_paths: dict[str, Path],
    output_root: Path,
) -> None:
    variant_counts = _variant_counts_by_level(variants)
    body = "\n".join(
        [
            "<h1>Final CADICA Sweep Benchmark Report</h1>",
            "<h2>1. Sweep Overview</h2>",
            "<p>This report summarizes binary weak-label agreement for stenosis detection. Severity-related tables and level-of-stenosis statistics are intentionally excluded.</p>",
            _html_table(
                [{"level": level, "benchmarked_variants": variant_counts.get(level, 0)} for level in LEVELS],
                ("level", "benchmarked_variants"),
            ),
            f"<p>{html.escape(_job_summary_sentence(benchmark_overview))}</p>",
            "<h2>2. Best Final Multi-View Result</h2>",
            _html_table([_best_result_row(best_multiview)], _best_result_columns()),
            _html_image(plot_paths["confusion_matrix"], output_root, "Best multi-view confusion matrix"),
            _html_image(plot_paths["precision_recall_scatter"], output_root, "Multi-view precision/recall scatter"),
            "<h2>3. Top Multi-View Variants</h2>",
            _html_table([_compact_metric_row(variant) for variant in top_multiview], _compact_metric_columns()),
            _html_link(table_paths["top_multiview_variants"], output_root, "Full CSV"),
            "<h2>4. Precision/Recall Tradeoff</h2>",
            _html_table([_compact_metric_row(variant) for variant in pareto_multiview[:10]], _compact_metric_columns()),
            _html_link(plot_paths["precision_recall_scatter"], output_root, "Scatter plot"),
            "<h2>5. Stage Comparison</h2>",
            "<p>Multi-view is the final case-level result. Frame and temporal rows are included only as stage comparisons.</p>",
            _html_table([_compact_metric_row(variant) for variant in best_by_level], _compact_metric_columns()),
            _html_image(plot_paths["best_by_level_comparison"], output_root, "Stage comparison"),
            "<h2>6. Parameter Sensitivity</h2>",
            f"<p>{html.escape(_sensitivity_summary(sensitivity_rows))}</p>",
            "<ul>",
            f"<li>{_html_link(plot_paths['temporal_parameter_heatmap'], output_root, 'Temporal heatmap')}</li>",
            f"<li>{_html_link(plot_paths['frame_threshold_heatmap'], output_root, 'Frame threshold heatmap')}</li>",
            f"<li>{_html_link(plot_paths['radius_parameter_heatmap'], output_root, 'Radius parameter heatmap')}</li>",
            "</ul>",
            "<h2>7. Files Generated</h2>",
            _html_file_list(table_paths, plot_paths, output_root),
        ]
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Final CADICA Sweep Benchmark Report</title>
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
    sweep_root: Path,
    benchmark_root: Path,
    output_root: Path,
    top_k: int,
    variants: list[VariantMetric],
    best_multiview: VariantMetric,
    best_by_level: list[VariantMetric],
    benchmark_overview: dict[str, Any],
    table_paths: dict[str, Path],
    plot_paths: dict[str, Path],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "title": "Final CADICA Sweep Benchmark Report",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sweep_root": str(sweep_root),
        "benchmark_root": str(benchmark_root),
        "output_root": str(output_root),
        "primary_level": "multiview",
        "top_k": top_k,
        "variant_counts_by_level": _variant_counts_by_level(variants),
        "benchmark_jobs": benchmark_overview,
        "best_multiview": best_multiview.to_table_row(),
        "best_by_level": [variant.to_table_row() for variant in best_by_level],
        "warnings": warnings,
        "tables": {name: str(path) for name, path in table_paths.items()},
        "plots": {name: str(path) for name, path in plot_paths.items()},
    }


def _summary_entries(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in ("completed", "skipped"):
        entries = payload.get(key)
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict):
                    yield entry


def _variant_from_summary_entry(entry: dict[str, Any], benchmark_root: Path) -> VariantMetric | None:
    level = str(entry.get("level") or "")
    if level not in LEVELS:
        return None
    frame_variant = entry.get("frame_variant") if isinstance(entry.get("frame_variant"), str) else None
    temporal_variant = entry.get("temporal_variant") if isinstance(entry.get("temporal_variant"), str) else None
    summary_path = _summary_path_from_entry(entry, benchmark_root, level, frame_variant, temporal_variant)
    if summary_path is None or not summary_path.is_file():
        return None
    variant = _variant_from_summary_path(summary_path, benchmark_root)
    if variant.row_count is not None or entry.get("row_count") is None:
        return variant
    return VariantMetric(
        level=variant.level,
        frame_variant=variant.frame_variant,
        temporal_variant=variant.temporal_variant,
        result_summary_path=variant.result_summary_path,
        row_count=_coerce_int(entry.get("row_count")),
        frame_params=variant.frame_params,
        temporal_params=variant.temporal_params,
        metrics=variant.metrics,
        rows_csv_path=variant.rows_csv_path,
    )


def _variant_from_summary_path(summary_path: Path, benchmark_root: Path) -> VariantMetric:
    level = _level_from_summary_path(summary_path)
    frame_variant, temporal_variant = _variants_from_summary_path(summary_path, benchmark_root, level)
    payload = _read_json(summary_path)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    row_count = _coerce_int(summary.get("total_predictions"))
    rows_csv_path = summary_path.with_name(ROW_FILENAMES[level])
    return VariantMetric(
        level=level,
        frame_variant=frame_variant,
        temporal_variant=temporal_variant,
        result_summary_path=summary_path,
        row_count=row_count,
        frame_params=parse_frame_variant(frame_variant),
        temporal_params=parse_temporal_variant(temporal_variant),
        metrics={metric: _coerce_metric(summary.get(metric)) for metric in BINARY_METRICS},
        rows_csv_path=rows_csv_path if rows_csv_path.is_file() else None,
    )


def _summary_path_from_entry(
    entry: dict[str, Any],
    benchmark_root: Path,
    level: str,
    frame_variant: str | None,
    temporal_variant: str | None,
) -> Path | None:
    expected = _expected_summary_path(benchmark_root, level, frame_variant, temporal_variant)
    if expected is not None and expected.is_file():
        return expected
    output_paths = entry.get("output_paths") if isinstance(entry.get("output_paths"), dict) else {}
    for raw_path in output_paths.values():
        if isinstance(raw_path, str) and raw_path.endswith(SUMMARY_FILENAMES[level]):
            resolved = _resolve_recorded_path(raw_path, benchmark_root)
            if resolved.is_file():
                return resolved
    output_root = entry.get("output_root")
    if isinstance(output_root, str):
        candidate = _resolve_recorded_path(str(Path(output_root) / SUMMARY_FILENAMES[level]), benchmark_root)
        if candidate.is_file():
            return candidate
    return None


def _expected_summary_path(
    benchmark_root: Path,
    level: str,
    frame_variant: str | None,
    temporal_variant: str | None,
) -> Path | None:
    if level == "frame" and frame_variant:
        return benchmark_root / "frame" / frame_variant / SUMMARY_FILENAMES[level]
    if level in {"temporal", "multiview"} and frame_variant and temporal_variant:
        return benchmark_root / level / frame_variant / temporal_variant / SUMMARY_FILENAMES[level]
    return None


def _resolve_recorded_path(raw_path: str, benchmark_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path
    direct_under_root = benchmark_root / path
    if direct_under_root.exists():
        return direct_under_root
    parts = path.parts
    if benchmark_root.name in parts:
        index = parts.index(benchmark_root.name)
        candidate = benchmark_root.joinpath(*parts[index + 1 :])
        if candidate.exists():
            return candidate
    return direct_under_root


def _discover_summary_paths(benchmark_root: Path) -> list[Path]:
    paths: list[Path] = []
    for filename in SUMMARY_FILENAMES.values():
        paths.extend(path for path in benchmark_root.rglob(filename) if path.is_file())
    return sorted(paths)


def _level_from_summary_path(summary_path: Path) -> str:
    for level, filename in SUMMARY_FILENAMES.items():
        if summary_path.name == filename:
            return level
    raise SweepReportError(f"Unsupported benchmark summary filename: {summary_path}")


def _variants_from_summary_path(summary_path: Path, benchmark_root: Path, level: str) -> tuple[str | None, str | None]:
    try:
        relative = summary_path.relative_to(benchmark_root)
    except ValueError:
        relative = summary_path
    parts = relative.parts
    if level == "frame" and len(parts) >= 3:
        return parts[-2], None
    if level in {"temporal", "multiview"} and len(parts) >= 4:
        return parts[-3], parts[-2]
    if level == "frame":
        return summary_path.parent.name, None
    return summary_path.parent.parent.name, summary_path.parent.name


def _parse_variant_parts(variant_name: str, parameter_names: tuple[str, ...]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    ordered_names = sorted(parameter_names, key=len, reverse=True)
    for part in variant_name.split("__"):
        for name in ordered_names:
            prefix = f"{name}_"
            if part.startswith(prefix):
                parsed[name] = part[len(prefix) :]
                break
    return parsed


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace("p", "."))
    except ValueError:
        return None


def _as_int(value: str | None) -> int | None:
    numeric = _as_float(value)
    return None if numeric is None else int(numeric)


def _coerce_metric(value: Any) -> int | float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if isfinite(value) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number.is_integer() and str(value).strip().isdigit():
        return int(number)
    return number if isfinite(number) else None


def _coerce_int(value: Any) -> int | None:
    metric = _coerce_metric(value)
    if metric is None:
        return None
    return int(metric)


def _safe_float(value: Any) -> float:
    metric = _coerce_metric(value)
    return 0.0 if metric is None else float(metric)


def _is_number(value: Any) -> bool:
    return _coerce_metric(value) is not None


def _ranking_float(value: Any) -> float:
    metric = _coerce_metric(value)
    return float("-inf") if metric is None else float(metric)


def _ranking_count(value: Any) -> int:
    metric = _coerce_metric(value)
    return 10**12 if metric is None else int(metric)


def _mean_metric(variants: list[VariantMetric], metric_name: str) -> float | None:
    values = [float(value) for value in (variant.metric(metric_name) for variant in variants) if value is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _max_metric(variants: list[VariantMetric], metric_name: str) -> float | None:
    values = [float(value) for value in (variant.metric(metric_name) for variant in variants) if value is not None]
    return None if not values else max(values)


def _best_by_level(variants: list[VariantMetric]) -> list[VariantMetric]:
    best: list[VariantMetric] = []
    for level in LEVELS:
        level_rows = [variant for variant in variants if variant.level == level]
        if level_rows:
            best.append(rank_variants(level_rows)[0])
    return best


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    field_list = list(fieldnames)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_list, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in field_list})


def _csv_value(value: Any) -> Any:
    return "" if value is None else value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise SweepReportError(f"Expected JSON object in benchmark summary: {path}")
    return payload


def _best_rows_have_split_output(variant: VariantMetric) -> bool:
    if variant.rows_csv_path is None or not variant.rows_csv_path.is_file():
        return False
    with variant.rows_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "split_by_coronary_side" not in (reader.fieldnames or []):
            return False
        for row in reader:
            if str(row.get("split_by_coronary_side", "")).strip().lower() == "true":
                return True
    return False


def _compact_variant_label(variant: VariantMetric) -> str:
    row = variant.to_table_row()
    parts = [
        f"rf={_format_param_value(row.get('radius_outside_fraction_threshold'))}",
        f"rs={_format_param_value(row.get('radius_min_outside_samples'))}",
        f"st={_format_param_value(row.get('stenosis_threshold'))}",
        f"ar={_format_param_value(row.get('average_radius_threshold'))}",
    ]
    if variant.temporal_variant:
        parts.extend(
            [
                f"sf={_format_param_value(row.get('min_supporting_frames'))}",
                f"pr={_format_param_value(row.get('min_persistence_ratio'))}",
            ]
        )
    return " ".join(parts)


def _format_param_value(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    numeric = _coerce_metric(value)
    if numeric is None:
        return str(value)
    if float(numeric).is_integer():
        return str(int(numeric))
    return f"{float(numeric):.2f}".rstrip("0").rstrip(".")


def _format_metric(value: Any) -> str:
    numeric = _coerce_metric(value)
    if numeric is None:
        return "n/a"
    return f"{float(numeric):.3f}"


def _plot_empty(output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14)
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _variant_counts_by_level(variants: list[VariantMetric]) -> dict[str, int]:
    return {level: sum(1 for variant in variants if variant.level == level) for level in LEVELS}


def _job_summary_sentence(overview: dict[str, Any]) -> str:
    completed = overview.get("completed_jobs")
    skipped = overview.get("skipped_jobs")
    failed = overview.get("failed_jobs")
    if completed is None and skipped is None and failed is None:
        return "Benchmark job counts were not available; summaries were discovered directly from disk."
    return f"Benchmark jobs: completed={completed}, skipped={skipped}, failed={failed}."


def _best_result_columns() -> tuple[str, ...]:
    return (
        *FRAME_PARAMS,
        *TEMPORAL_PARAMS,
        "TP",
        "FP",
        "TN",
        "FN",
        "precision",
        "recall",
        "f1",
        "specificity",
        "balanced_accuracy",
        "total_evaluated",
    )


def _best_result_row(variant: VariantMetric) -> dict[str, Any]:
    row = variant.to_table_row()
    return {column: row.get(column) for column in _best_result_columns()}


def _compact_metric_columns() -> tuple[str, ...]:
    return (
        "level",
        "variant",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "TP",
        "FP",
        "TN",
        "FN",
    )


def _compact_metric_row(variant: VariantMetric) -> dict[str, Any]:
    return {
        "level": variant.level,
        "variant": _compact_variant_label(variant),
        "precision": _format_metric(variant.metric("precision")),
        "recall": _format_metric(variant.metric("recall")),
        "f1": _format_metric(variant.metric("f1")),
        "balanced_accuracy": _format_metric(variant.metric("balanced_accuracy")),
        "TP": variant.metric("TP"),
        "FP": variant.metric("FP"),
        "TN": variant.metric("TN"),
        "FN": variant.metric("FN"),
    }


def _markdown_table(rows: list[dict[str, Any]], columns: Iterable[str]) -> str:
    column_list = list(columns)
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(column_list) + " |"
    separator = "| " + " | ".join("---" for _ in column_list) + " |"
    body = [
        "| " + " | ".join(_markdown_value(row.get(column)) for column in column_list) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return _format_metric(value)
    return "" if value is None else str(value)


def _html_table(rows: list[dict[str, Any]], columns: Iterable[str]) -> str:
    column_list = list(columns)
    if not rows:
        return "<p><em>No rows.</em></p>"
    header = "".join(f"<th>{html.escape(column)}</th>" for column in column_list)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(_markdown_value(row.get(column)))}</td>" for column in column_list)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _html_image(path: Path, output_root: Path, alt: str) -> str:
    relative = html.escape(_relative_path(path, output_root))
    return f'<img src="{relative}" alt="{html.escape(alt)}">'


def _html_link(path: Path, output_root: Path, label: str) -> str:
    relative = html.escape(_relative_path(path, output_root))
    return f'<a href="{relative}">{html.escape(label)}</a>'


def _html_file_list(table_paths: dict[str, Path], plot_paths: dict[str, Path], output_root: Path) -> str:
    items = []
    for path in (*table_paths.values(), *plot_paths.values()):
        relative = _relative_path(path, output_root)
        items.append(f'<li><a href="{html.escape(relative)}">{html.escape(relative)}</a></li>')
    return "<ul>" + "".join(items) + "</ul>"


def _generated_files_markdown(table_paths: dict[str, Path], plot_paths: dict[str, Path], output_root: Path) -> list[str]:
    return [
        f"- [{_relative_path(path, output_root)}]({_relative_path(path, output_root)})"
        for path in (*table_paths.values(), *plot_paths.values())
    ]


def _relative_path(path: Path, output_root: Path) -> str:
    try:
        return path.relative_to(output_root).as_posix()
    except ValueError:
        return str(path)


def _sensitivity_summary(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No parameter sensitivity rows were available."
    fragments: list[str] = []
    for parameter in (*FRAME_PARAMS, *TEMPORAL_PARAMS):
        parameter_rows = [row for row in rows if row.get("parameter") == parameter and row.get("max_f1") is not None]
        if not parameter_rows:
            continue
        best = max(parameter_rows, key=lambda row: float(row["max_f1"]))
        fragments.append(f"{parameter}={_format_param_value(best.get('value'))} had the best max F1 ({_format_metric(best.get('max_f1'))})")
    if not fragments:
        return "No parameter values had numeric F1 values."
    return "; ".join(fragments) + "."


if __name__ == "__main__":
    raise SystemExit(main())
