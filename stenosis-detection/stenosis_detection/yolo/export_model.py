from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from .utils import YoloDependencyError, compact_kwargs, load_yolo_class


EXPORT_DIR = Path("outputs/yolo_exports")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a trained YOLO detection model.")
    parser.add_argument("--weights", required=True, type=Path, help="Path to trained weights, usually best.pt.")
    parser.add_argument(
        "--format",
        default="onnx",
        choices=("onnx", "torchscript", "openvino"),
        help="Export format.",
    )
    parser.add_argument("--imgsz", type=int, help="Export image size.")
    parser.add_argument("--device", help="Device string, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 export when supported.")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX export when supported.")
    parser.add_argument("--output-dir", type=Path, default=EXPORT_DIR, help="Directory where exported artifacts are copied.")
    return parser


def exported_paths(exported: Any) -> list[Path]:
    if exported is None:
        return []
    if isinstance(exported, (list, tuple, set)):
        return [Path(path) for path in exported]
    return [Path(exported)]


def increment_export_path(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(2, 10_000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find an available export path for {path}")


def export_name(weights: Path, source: Path, export_format: str) -> str:
    if weights.parent.name == "weights":
        run_name = weights.parent.parent.name
    else:
        run_name = weights.stem

    if source.is_dir():
        return f"{run_name}_{export_format}"
    return f"{run_name}{source.suffix}"


def copy_export_to_output(source: Path, weights: Path, export_format: str, output_dir: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Ultralytics reported an export path that does not exist: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = increment_export_path(output_dir / export_name(weights, source, export_format))

    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)

    return destination


def run_export(args: argparse.Namespace) -> list[Path]:
    YOLO = load_yolo_class()
    model = YOLO(str(args.weights))
    kwargs = compact_kwargs(
        {
            "format": args.format,
            "imgsz": args.imgsz,
            "device": args.device,
            "half": args.half,
            "simplify": args.simplify,
        }
    )
    try:
        exported = model.export(**kwargs)
    except (TypeError, SyntaxError) as exc:
        raise ValueError("Ultralytics rejected one or more export arguments.") from exc

    paths = exported_paths(exported)
    if not paths:
        raise ValueError("Ultralytics did not return an exported model path.")
    return [copy_export_to_output(path, args.weights, args.format, args.output_dir) for path in paths]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        saved_paths = run_export(args)
    except (OSError, ValueError, YoloDependencyError) as exc:
        raise SystemExit(f"YOLO export failed: {exc}") from exc

    for path in saved_paths:
        print(f"Exported model: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
