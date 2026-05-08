from __future__ import annotations

from pathlib import Path
from typing import Any


class YoloDependencyError(RuntimeError):
    """Raised when optional YOLO dependencies are not installed."""


def load_yaml_module() -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise YoloDependencyError("Missing dependency: pyyaml. Run `pip install -r requirements.txt`.") from exc
    return yaml


def load_yolo_class() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise YoloDependencyError("Missing dependency: ultralytics. Run `pip install -r requirements.txt`.") from exc
    return YOLO


def compact_kwargs(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def increment_path(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(2, 10_000):
        candidate = Path(f"{path}{index}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find an available path for {path}")


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    return str(value)
