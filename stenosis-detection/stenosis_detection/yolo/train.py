from __future__ import annotations

import argparse
from typing import Any

from .config import (
    format_config,
    load_config,
    merge_training_config,
    parse_resume,
    parse_unknown_overrides,
    prepare_training_args,
    save_resolved_config,
    save_to_actual_run_dir,
    set_deterministic_seed,
)
from .utils import YoloDependencyError, load_yolo_class


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an Ultralytics YOLO detection model. Unknown --key value "
            "pairs are accepted as additional Ultralytics training overrides."
        )
    )
    parser.add_argument("--config", help="YAML config with default training values.")
    parser.add_argument("--data", help="Path to dataset data.yaml.")
    parser.add_argument("--model", help="Ultralytics-compatible model, e.g. yolov8x.pt or yolo11x.pt.")
    parser.add_argument("--imgsz", type=int, help="Training image size.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, help="Batch size. Use -1 for Ultralytics auto-batch.")
    parser.add_argument("--device", help="Device string, e.g. 0, 0,1, cpu, or mps.")
    parser.add_argument("--project", help="Project directory for runs.")
    parser.add_argument("--name", help="Run name.")
    parser.add_argument("--workers", type=int, help="Data loader workers.")
    parser.add_argument("--patience", type=int, help="Early stopping patience.")
    parser.add_argument("--optimizer", help="Optimizer, e.g. auto, AdamW, Adam, or SGD.")
    parser.add_argument("--lr0", type=float, help="Initial learning rate.")
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, help="Weight decay.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        type=parse_resume,
        help="Resume training. Optionally pass true/false or a checkpoint path.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    return build_parser().parse_known_args(argv)


def run_training(model_name: str, train_args: dict[str, Any]) -> Any:
    YOLO = load_yolo_class()
    model = YOLO(model_name)
    try:
        model.train(**train_args)
    except (TypeError, SyntaxError) as exc:
        raise ValueError(
            "Ultralytics rejected one or more training arguments. "
            "Check the resolved config and remove or rename unsupported parameters."
        ) from exc
    return model


def main(argv: list[str] | None = None) -> int:
    args, unknown_tokens = parse_args(argv)
    try:
        config = load_config(args.config)
        extra_overrides = parse_unknown_overrides(unknown_tokens)
        resolved_config = merge_training_config(config, args, extra_overrides)
        model_name = str(resolved_config["model"])
        train_args, run_dir = prepare_training_args(resolved_config)

        set_deterministic_seed(resolved_config.get("seed"))
        print("Resolved training configuration:")
        print(format_config(resolved_config))

        config_path = save_resolved_config(resolved_config, run_dir)
        print(f"Saved resolved config: {config_path}")

        model = run_training(model_name, train_args)
        actual_config_path = save_to_actual_run_dir(config_path, model)
        if actual_config_path is not None and actual_config_path != config_path:
            print(f"Saved resolved config to actual run folder: {actual_config_path}")
    except (OSError, ValueError, YoloDependencyError) as exc:
        raise SystemExit(f"YOLO training failed: {exc}") from exc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
