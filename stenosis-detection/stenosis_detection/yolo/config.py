from __future__ import annotations

import os
import random
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping

from .utils import increment_path, load_yaml_module


CORE_TRAIN_ARGS = {
    "data",
    "model",
    "imgsz",
    "epochs",
    "batch",
    "device",
    "project",
    "name",
    "workers",
    "patience",
    "optimizer",
    "lr0",
    "weight_decay",
    "seed",
    "resume",
}

DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "model": "yolov8x.pt",
    "imgsz": 1024,
    "epochs": 100,
    "batch": 8,
    "device": None,
    "project": "runs/yolo_stenosis",
    "name": "stenosis_train",
    "workers": 8,
    "patience": 30,
    "optimizer": "auto",
    "lr0": 0.001,
    "weight_decay": 0.0005,
    "seed": 42,
    "resume": False,
    "pretrained": True,
    "deterministic": True,
    "task": "detect",
}

RESOLVED_TRAIN_CONFIG_NAME = "resolved_train_config.yaml"


def parse_resume(value: str | None) -> bool | str:
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return value


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    yaml = load_yaml_module()
    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{resolved_path} must contain a YAML mapping.")
    return dict(data)


def parse_unknown_overrides(tokens: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    index = 0
    yaml = load_yaml_module() if tokens else None

    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected argument `{token}`. Extra overrides must use --key value syntax.")

        key_value = token[2:]
        if not key_value:
            raise ValueError("Empty override key is not allowed.")

        if "=" in key_value:
            key, raw_value = key_value.split("=", 1)
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            key = key_value
            raw_value = tokens[index + 1]
            index += 1
        else:
            key = key_value
            raw_value = "true"

        key = key.replace("-", "_")
        if not key:
            raise ValueError("Empty override key is not allowed.")
        overrides[key] = yaml.safe_load(raw_value) if yaml is not None else raw_value
        index += 1

    return overrides


def merge_training_config(
    config: Mapping[str, Any],
    cli_values: Namespace | Mapping[str, Any],
    extra_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(DEFAULT_TRAIN_CONFIG)
    merged.update(dict(config))

    values = vars(cli_values) if isinstance(cli_values, Namespace) else dict(cli_values)
    for key in CORE_TRAIN_ARGS:
        value = values.get(key)
        if value is not None:
            merged[key] = value

    if extra_overrides is not None:
        merged.update(dict(extra_overrides))

    merged["task"] = "detect"
    merged["project"] = str(merged.get("project") or DEFAULT_TRAIN_CONFIG["project"])

    data = merged.get("data")
    if data is None:
        raise ValueError("`--data` is required unless it is provided in the config.")
    merged["data"] = str(data)

    model = merged.get("model")
    if not model:
        raise ValueError("`--model` or config value `model` is required.")
    merged["model"] = str(model)

    return {key: value for key, value in merged.items() if value is not None}


def set_deterministic_seed(seed: int | None) -> None:
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def expected_run_dir(config: Mapping[str, Any]) -> Path:
    project = Path(str(config.get("project", DEFAULT_TRAIN_CONFIG["project"])))
    name = str(config.get("name", "train"))
    run_dir = project / name
    if bool(config.get("exist_ok", False)) or config.get("resume"):
        return run_dir
    return increment_path(run_dir)


def run_dir_from_resume(resume: Any) -> Path | None:
    if not isinstance(resume, str):
        return None

    checkpoint = Path(resume)
    if not checkpoint.exists():
        return None
    if checkpoint.parent.name == "weights":
        return checkpoint.parent.parent
    return checkpoint.parent


def prepare_training_args(resolved_config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    train_args = {key: value for key, value in resolved_config.items() if key != "model"}
    resume = train_args.get("resume")

    if resume:
        configured_run_dir = Path(str(train_args.get("project", DEFAULT_TRAIN_CONFIG["project"]))) / str(
            train_args.get("name", "train")
        )
        run_dir = run_dir_from_resume(resume) or configured_run_dir
        return train_args, run_dir

    run_dir = expected_run_dir(train_args)
    train_args["project"] = str(run_dir.parent)
    train_args["name"] = run_dir.name
    train_args["exist_ok"] = True
    resolved_config["project"] = train_args["project"]
    resolved_config["name"] = train_args["name"]
    resolved_config["exist_ok"] = True
    return train_args, run_dir


def save_resolved_config(config: Mapping[str, Any], run_dir: str | Path) -> Path:
    yaml = load_yaml_module()
    resolved_run_dir = Path(run_dir)
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    output_path = resolved_run_dir / RESOLVED_TRAIN_CONFIG_NAME
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(dict(config), file, sort_keys=True)
    return output_path


def save_to_actual_run_dir(config_path: Path, model: Any) -> Path | None:
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir is None:
        return None

    actual_path = Path(save_dir) / RESOLVED_TRAIN_CONFIG_NAME
    if actual_path.resolve() == config_path.resolve():
        return actual_path

    actual_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, actual_path)
    return actual_path


def format_config(config: Mapping[str, Any]) -> str:
    yaml = load_yaml_module()
    return yaml.safe_dump(dict(config), sort_keys=True).strip()
