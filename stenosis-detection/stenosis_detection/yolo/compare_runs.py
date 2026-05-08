from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


PLOT_DIR = Path("outputs/yolo_comparisons")

METRIC_ALIASES = {
    "precision": ("metrics/precision(B)", "metrics/precision", "precision"),
    "recall": ("metrics/recall(B)", "metrics/recall", "recall"),
    "mAP50": ("metrics/mAP50(B)", "metrics/mAP50", "mAP50"),
    "mAP50-95": ("metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95"),
    "train_box_loss": ("train/box_loss",),
    "train_cls_loss": ("train/cls_loss",),
    "train_dfl_loss": ("train/dfl_loss",),
    "val_box_loss": ("val/box_loss",),
    "val_cls_loss": ("val/cls_loss",),
    "val_dfl_loss": ("val/dfl_loss",),
}

OUTPUT_COLUMNS = [
    "rank",
    "run",
    "best_epoch",
    "precision",
    "recall",
    "mAP50",
    "mAP50-95",
    "train_box_loss",
    "train_cls_loss",
    "train_dfl_loss",
    "val_box_loss",
    "val_cls_loss",
    "val_dfl_loss",
    "best_pt",
    "best_pt_exists",
    "results_csv",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare YOLO training runs and identify candidate best models.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/yolo_stenosis"), help="Directory containing YOLO run folders.")
    parser.add_argument("--output", type=Path, default=Path("outputs/yolo_run_comparison.csv"), help="CSV summary output path.")
    return parser


def clean_row(row: dict[str, Any]) -> dict[str, str]:
    return {key.strip(): str(value).strip() for key, value in row.items() if key is not None}


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric_value(row: dict[str, str], metric_name: str) -> float | None:
    for column in METRIC_ALIASES[metric_name]:
        if column in row:
            return to_float(row[column])
    return None


def epoch_value(row: dict[str, str]) -> int | None:
    value = to_float(row.get("epoch"))
    return int(value) if value is not None else None


def sort_score(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        row["mAP50-95"] if row["mAP50-95"] is not None else float("-inf"),
        row["recall"] if row["recall"] is not None else float("-inf"),
        row["mAP50"] if row["mAP50"] is not None else float("-inf"),
    )


def read_results_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return [clean_row(row) for row in csv.DictReader(file)]


def best_row_for_results(results_path: Path, runs_dir: Path) -> dict[str, Any] | None:
    rows = read_results_csv(results_path)
    if not rows:
        return None

    candidates: list[dict[str, Any]] = []
    for row in rows:
        candidate = {
            "best_epoch": epoch_value(row),
            "precision": metric_value(row, "precision"),
            "recall": metric_value(row, "recall"),
            "mAP50": metric_value(row, "mAP50"),
            "mAP50-95": metric_value(row, "mAP50-95"),
            "train_box_loss": metric_value(row, "train_box_loss"),
            "train_cls_loss": metric_value(row, "train_cls_loss"),
            "train_dfl_loss": metric_value(row, "train_dfl_loss"),
            "val_box_loss": metric_value(row, "val_box_loss"),
            "val_cls_loss": metric_value(row, "val_cls_loss"),
            "val_dfl_loss": metric_value(row, "val_dfl_loss"),
        }
        if candidate["mAP50-95"] is None and candidate["recall"] is None and candidate["mAP50"] is None:
            continue
        candidates.append(candidate)

    if not candidates:
        return None

    best = max(candidates, key=sort_score)
    run_dir = results_path.parent
    best_pt = run_dir / "weights" / "best.pt"
    best.update(
        {
            "run": str(run_dir.relative_to(runs_dir)),
            "best_pt": str(best_pt),
            "best_pt_exists": best_pt.exists(),
            "results_csv": str(results_path),
        }
    )
    return best


def collect_run_summaries(runs_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for results_path in sorted(runs_dir.glob("**/results.csv")):
        summary = best_row_for_results(results_path, runs_dir)
        if summary is not None:
            summaries.append(summary)

    summaries.sort(key=sort_score, reverse=True)
    for rank, row in enumerate(summaries, start=1):
        row["rank"] = rank
    return summaries


def format_csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def save_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_csv_value(row.get(column)) for column in OUTPUT_COLUMNS})


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "n/a"
    return str(value)


def print_top_runs(rows: list[dict[str, Any]], limit: int = 5) -> None:
    print(f"Top {min(limit, len(rows))} runs")
    print("rank | run | epoch | recall | mAP50-95 | mAP50 | precision | best.pt")
    print("-" * 110)
    for row in rows[:limit]:
        print(
            f"{row['rank']} | {row['run']} | {fmt(row['best_epoch'])} | "
            f"{fmt(row['recall'])} | {fmt(row['mAP50-95'])} | {fmt(row['mAP50'])} | "
            f"{fmt(row['precision'])} | {row['best_pt']}"
        )


def save_bar_plot(rows: list[dict[str, Any]], metric: str, output_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plot output.
        return

    plot_rows = [row for row in rows if row.get(metric) is not None][:10]
    if not plot_rows:
        return

    labels = [row["run"] for row in plot_rows][::-1]
    values = [row[metric] for row in plot_rows][::-1]
    height = max(4.0, 0.45 * len(plot_rows))

    _, axis = plt.subplots(figsize=(11, height))
    axis.barh(labels, values)
    axis.set_xlabel(metric)
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_recall_map_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plot output.
        return

    plot_rows = [row for row in rows if row.get("recall") is not None and row.get("mAP50-95") is not None][:20]
    if not plot_rows:
        return

    _, axis = plt.subplots(figsize=(9, 6))
    axis.scatter([row["recall"] for row in plot_rows], [row["mAP50-95"] for row in plot_rows])
    for row in plot_rows[:10]:
        axis.annotate(str(row["rank"]), (row["recall"], row["mAP50-95"]), textcoords="offset points", xytext=(5, 5))
    axis.set_xlabel("recall")
    axis.set_ylabel("mAP50-95")
    axis.set_title("Recall vs mAP50-95")
    axis.grid(alpha=0.25)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_comparison_plots(rows: list[dict[str, Any]], plot_dir: Path = PLOT_DIR) -> list[Path]:
    paths = [
        plot_dir / "top_map50_95.png",
        plot_dir / "top_recall.png",
        plot_dir / "recall_vs_map50_95.png",
    ]
    save_bar_plot(rows, "mAP50-95", paths[0], "Top runs by mAP50-95")
    save_bar_plot(rows, "recall", paths[1], "Top runs by recall")
    save_recall_map_plot(rows, paths[2])
    return [path for path in paths if path.exists()]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.runs_dir.exists():
        raise SystemExit(f"Runs directory not found: {args.runs_dir}")

    rows = collect_run_summaries(args.runs_dir)
    if not rows:
        raise SystemExit(f"No usable results.csv files found under {args.runs_dir}")

    save_summary_csv(rows, args.output)
    plot_paths = save_comparison_plots(rows)

    print_top_runs(rows)
    print("\nSorted by mAP50-95, then recall, then mAP50.")
    print("Recall is shown explicitly because missed stenosis detections are costly.")
    print(f"Saved CSV: {args.output}")
    for path in plot_paths:
        print(f"Saved plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
