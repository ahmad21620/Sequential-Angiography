from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import load_yaml_module


EXPECTED_CLASS_NAME = "Stenosis"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
REQUIRED_YAML_KEYS = ("train", "val", "test", "nc", "names")


@dataclass(frozen=True)
class Box:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class SplitReport:
    name: str
    image_dir: Path
    label_dir: Path
    image_count: int = 0
    label_count: int = 0
    empty_label_count: int = 0
    missing_label_files: list[Path] = field(default_factory=list)
    orphan_label_files: list[Path] = field(default_factory=list)
    invalid_lines: list[str] = field(default_factory=list)
    valid_box_count: int = 0

    @property
    def failures(self) -> list[str]:
        failures: list[str] = []
        if not self.image_dir.is_dir():
            failures.append(f"{self.name}: image folder is missing: {self.image_dir}")
        if not self.label_dir.is_dir():
            failures.append(f"{self.name}: label folder is missing: {self.label_dir}")
        if self.image_dir.is_dir() and self.image_count == 0:
            failures.append(f"{self.name}: image folder contains no supported images")
        if self.label_dir.is_dir() and self.label_count == 0:
            failures.append(f"{self.name}: label folder contains no .txt files")
        if self.missing_label_files:
            failures.append(f"{self.name}: {len(self.missing_label_files)} images are missing label files")
        if self.invalid_lines:
            failures.append(f"{self.name}: {len(self.invalid_lines)} invalid label lines")
        return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check a YOLO detection dataset for common format issues.")
    parser.add_argument("--data", required=True, type=Path, help="Path to dataset data.yaml.")
    return parser


def load_yaml(path: Path) -> dict[str, Any]:
    yaml = load_yaml_module()
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError("data.yaml must contain a YAML mapping.")
    return loaded


def parse_names(value: Any) -> list[str] | None:
    if isinstance(value, list):
        return [str(name) for name in value]
    if isinstance(value, dict):
        parsed: list[tuple[int, str]] = []
        for key, name in value.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                return None
            parsed.append((index, str(name)))
        return [name for _, name in sorted(parsed)]
    return None


def resolve_dataset_path(value: Any, *, base_dir: Path, dataset_root: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    root_candidate = (dataset_root / path).resolve()
    base_candidate = (base_dir / path).resolve()
    if root_candidate.exists() or not base_candidate.exists():
        return root_candidate
    return base_candidate


def image_dir_from_split_path(path: Path) -> Path:
    return path if path.name == "images" else path / "images"


def image_files(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        return []
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def label_files(label_dir: Path) -> list[Path]:
    if not label_dir.is_dir():
        return []
    return sorted(path for path in label_dir.glob("*.txt") if path.is_file())


def parse_label_line(line: str, label_path: Path, line_number: int, class_count: int) -> Box | str:
    parts = line.split()
    if len(parts) != 5:
        return f"{label_path}:{line_number} expected 5 columns, found {len(parts)}"
    try:
        class_id = int(parts[0])
        x_center, y_center, width, height = (float(value) for value in parts[1:])
    except ValueError:
        return f"{label_path}:{line_number} class and box values must be numeric"
    box = Box(class_id, x_center, y_center, width, height)
    if class_id < 0 or class_id >= class_count:
        return f"{label_path}:{line_number} class id {class_id} outside 0..{class_count - 1}"
    if not has_valid_normalized_coordinates(box):
        return f"{label_path}:{line_number} invalid normalized box ({x_center:g}, {y_center:g}, {width:g}, {height:g})"
    return box


def has_valid_normalized_coordinates(box: Box) -> bool:
    x_min = box.x_center - box.width / 2.0
    x_max = box.x_center + box.width / 2.0
    y_min = box.y_center - box.height / 2.0
    y_max = box.y_center + box.height / 2.0
    return (
        0.0 <= box.x_center <= 1.0
        and 0.0 <= box.y_center <= 1.0
        and 0.0 < box.width <= 1.0
        and 0.0 < box.height <= 1.0
        and 0.0 <= x_min <= 1.0
        and 0.0 <= x_max <= 1.0
        and 0.0 <= y_min <= 1.0
        and 0.0 <= y_max <= 1.0
    )


def validate_split(split_name: str, split_path: Path, class_count: int) -> SplitReport:
    image_dir = image_dir_from_split_path(split_path)
    label_dir = image_dir.parent / "labels"
    images = image_files(image_dir)
    labels = label_files(label_dir)
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    report = SplitReport(
        name=split_name,
        image_dir=image_dir,
        label_dir=label_dir,
        image_count=len(images),
        label_count=len(labels),
        missing_label_files=[label_dir / f"{path.stem}.txt" for path in images if path.stem not in label_stems],
        orphan_label_files=[path for path in labels if path.stem not in image_stems],
    )

    for label_path in labels:
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            report.empty_label_count += 1
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            parsed = parse_label_line(line, label_path, line_number, class_count)
            if isinstance(parsed, str):
                report.invalid_lines.append(parsed)
            else:
                report.valid_box_count += 1
    return report


def check_dataset(data_yaml: Path) -> tuple[list[SplitReport], list[str], list[str]]:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml does not exist: {data_yaml}")
    data = load_yaml(data_yaml)
    failures: list[str] = []
    warnings: list[str] = []

    missing_keys = [key for key in REQUIRED_YAML_KEYS if key not in data]
    failures.extend(f"data.yaml is missing required key `{key}`" for key in missing_keys)
    if "val" in missing_keys and "valid" in data:
        failures.append("data.yaml has `valid`, but Ultralytics expects the validation key to be `val`")

    names = parse_names(data.get("names"))
    if names is None:
        failures.append("data.yaml `names` must be a list or numeric-indexed mapping")
        names = []

    try:
        class_count = int(data.get("nc"))
    except (TypeError, ValueError):
        failures.append("data.yaml `nc` must be an integer")
        class_count = 0
    if class_count != len(names):
        failures.append(f"data.yaml `nc` is {class_count}, but `names` contains {len(names)} classes")
    if not any(EXPECTED_CLASS_NAME.lower() in name.lower() for name in names):
        warnings.append(f"data.yaml `names` does not include `{EXPECTED_CLASS_NAME}`")

    base_dir = data_yaml.parent.resolve()
    dataset_root = resolve_dataset_path(data.get("path", "."), base_dir=base_dir, dataset_root=base_dir)
    reports: list[SplitReport] = []
    for split_name, yaml_key in (("train", "train"), ("val", "val"), ("test", "test")):
        if yaml_key not in data:
            continue
        split_path = resolve_dataset_path(data[yaml_key], base_dir=base_dir, dataset_root=dataset_root)
        report = validate_split(split_name, split_path, class_count)
        reports.append(report)
        failures.extend(report.failures)
        if report.empty_label_count:
            warnings.append(f"{split_name}: {report.empty_label_count} empty label files found")
        if report.orphan_label_files:
            warnings.append(f"{split_name}: {len(report.orphan_label_files)} orphan label files found")
    return reports, failures, warnings


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        reports, failures, warnings = check_dataset(args.data.resolve())
    except (OSError, ValueError) as exc:
        print(f"FINAL SUMMARY: FAIL\nfailures: 1\n  - {exc}\nwarnings: 0")
        return 1

    print(f"data.yaml: {args.data.resolve()}")
    for report in reports:
        print(f"\n[{report.name}]")
        print(f"image folder: {report.image_dir}")
        print(f"label folder: {report.label_dir}")
        print(f"images: {report.image_count}")
        print(f"label files: {report.label_count}")
        print(f"empty label files: {report.empty_label_count}")
        print(f"valid boxes: {report.valid_box_count}")

    status = "FAIL" if failures else "WARNING" if warnings else "PASS"
    print(f"\nFINAL SUMMARY: {status}")
    print(f"failures: {len(failures)}")
    for failure in failures[:10]:
        print(f"  - {failure}")
    print(f"warnings: {len(warnings)}")
    for warning in warnings[:10]:
        print(f"  - {warning}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
