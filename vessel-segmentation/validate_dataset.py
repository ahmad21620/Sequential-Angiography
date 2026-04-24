from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drive_seg.dataset import format_validation_report, validate_dataset_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an explicit vessel-segmentation dataset root with train/val/test-style splits."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root containing split folders such as train/, val/, and test/.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Split names to validate.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = validate_dataset_root(args.dataset_root.resolve(), splits=args.splits)
    print(format_validation_report(report))
    raise SystemExit(0 if report.is_valid else 1)


if __name__ == "__main__":
    main()
