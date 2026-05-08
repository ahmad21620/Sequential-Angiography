from __future__ import annotations

import argparse
from typing import Any

from .utils import YoloDependencyError, compact_kwargs, load_yolo_class


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLO predictions for visual/manual review.")
    parser.add_argument("--weights", required=True, help="Path to trained weights, usually best.pt.")
    parser.add_argument("--source", required=True, help="Folder, image, video, webcam index, or URL.")
    parser.add_argument("--imgsz", type=int, help="Prediction image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS.")
    parser.add_argument("--device", help="Device string, e.g. 0, 0,1, cpu, or mps.")
    parser.add_argument("--project", default="runs/yolo_stenosis", help="Project directory for prediction runs.")
    parser.add_argument("--name", default="predict_review", help="Prediction run name.")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO txt predictions.")
    parser.add_argument("--save-conf", action="store_true", help="Include confidence in saved txt predictions.")
    return parser


def run_prediction_review(args: argparse.Namespace) -> Any:
    YOLO = load_yolo_class()
    model = YOLO(args.weights)
    kwargs = compact_kwargs(
        {
            "source": args.source,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
            "project": args.project,
            "name": args.name,
            "save": True,
            "save_txt": args.save_txt,
            "save_conf": args.save_conf,
            "task": "detect",
        }
    )
    return model.predict(**kwargs)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        results = run_prediction_review(args)
    except (OSError, ValueError, YoloDependencyError) as exc:
        raise SystemExit(f"YOLO prediction review failed: {exc}") from exc

    save_dir = getattr(results[0], "save_dir", None) if results else None
    if save_dir is not None:
        print(f"Saved predictions: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
