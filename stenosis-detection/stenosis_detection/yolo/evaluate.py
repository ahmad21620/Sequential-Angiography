from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .utils import YoloDependencyError, compact_kwargs, json_safe, load_yolo_class


METRIC_KEYS = {
    "precision": ("metrics/precision(B)", "precision"),
    "recall": ("metrics/recall(B)", "recall"),
    "mAP50": ("metrics/mAP50(B)", "mAP50"),
    "mAP50-95": ("metrics/mAP50-95(B)", "mAP50-95"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO detection model.")
    parser.add_argument("--weights", required=True, help="Path to trained weights, usually best.pt.")
    parser.add_argument("--data", required=True, type=Path, help="Path to dataset data.yaml.")
    parser.add_argument("--split", default="test", choices=("val", "test"), help="Dataset split to evaluate.")
    parser.add_argument("--imgsz", type=int, help="Evaluation image size.")
    parser.add_argument("--batch", type=int, help="Batch size.")
    parser.add_argument("--device", help="Device string, e.g. 0, 0,1, cpu, or mps.")
    parser.add_argument("--conf", type=float, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, help="IoU threshold for NMS.")
    parser.add_argument("--project", default="runs/yolo_stenosis", help="Project directory for evaluation runs.")
    parser.add_argument("--name", help="Evaluation run name. Defaults to evaluate_<split>.")
    return parser


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def results_dict(metrics: Any) -> dict[str, Any]:
    values = getattr(metrics, "results_dict", None)
    return values if isinstance(values, dict) else {}


def metric_from_results(metrics: Any, result_key: str, box_attr: str) -> float | None:
    values = results_dict(metrics)
    if result_key in values:
        return as_float(values[result_key])

    box_metrics = getattr(metrics, "box", None)
    if box_metrics is not None and hasattr(box_metrics, box_attr):
        return as_float(getattr(box_metrics, box_attr))

    return None


def extract_metrics(metrics: Any) -> dict[str, float | None]:
    return {
        display_name: metric_from_results(metrics, result_key, box_attr)
        for display_name, (result_key, box_attr) in METRIC_KEYS.items()
    }


def save_dir_for(metrics: Any, project: str, name: str) -> Path:
    save_dir = getattr(metrics, "save_dir", None)
    if save_dir is not None:
        return Path(save_dir)
    return Path(project) / name


def save_metrics(save_dir: Path, summary: dict[str, Any], raw_results: dict[str, Any]) -> tuple[Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "metrics.csv"
    json_path = save_dir / "metrics.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    payload = dict(summary)
    payload["raw_results"] = json_safe(raw_results)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")

    return csv_path, json_path


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    run_name = args.name or f"evaluate_{args.split}"
    YOLO = load_yolo_class()
    model = YOLO(args.weights)
    kwargs = compact_kwargs(
        {
            "data": str(args.data),
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "conf": args.conf,
            "iou": args.iou,
            "project": args.project,
            "name": run_name,
            "plots": True,
            "task": "detect",
        }
    )
    metrics = model.val(**kwargs)
    save_dir = save_dir_for(metrics, args.project, run_name)
    summary: dict[str, Any] = {
        "weights": args.weights,
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "project": args.project,
        "name": run_name,
        "save_dir": str(save_dir),
        **extract_metrics(metrics),
    }
    csv_path, json_path = save_metrics(save_dir, summary, results_dict(metrics))
    summary["metrics_csv"] = str(csv_path)
    summary["metrics_json"] = str(json_path)
    return summary


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_evaluation(args)
    except (OSError, ValueError, YoloDependencyError) as exc:
        raise SystemExit(f"YOLO evaluation failed: {exc}") from exc

    print("Evaluation metrics:")
    print(f"  precision: {format_metric(summary['precision'])}")
    print(f"  recall: {format_metric(summary['recall'])}")
    print(f"  mAP50: {format_metric(summary['mAP50'])}")
    print(f"  mAP50-95: {format_metric(summary['mAP50-95'])}")
    print(f"Saved metrics CSV: {summary['metrics_csv']}")
    print(f"Saved metrics JSON: {summary['metrics_json']}")
    print(f"Ultralytics validation artifacts: {summary['save_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
