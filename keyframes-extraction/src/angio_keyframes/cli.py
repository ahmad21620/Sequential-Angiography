from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from angio_keyframes.backends import BACKEND_CHOICES
from angio_keyframes.pipeline import extract_keyframes_from_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract the strongest vessel-fill keyframe window from coronary angiogram frame directories."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="A single frames directory or a dataset root that contains frames directories.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=6,
        help="Number of keyframes to keep per sequence. Default: 6.",
    )
    parser.add_argument(
        "--baseline-frames",
        type=int,
        default=3,
        help=(
            "Number of initial frames used to build the pre-contrast baseline. "
            "Default: 3."
        ),
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help=(
            "Odd centered moving-average window used to smooth frame scores over time. "
            "Default: 5."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of frame directories to process in parallel for the cpu backend. Default: 1.",
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default="cpu",
        help="Image-processing backend to use. Default: cpu.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Root directory for the mirrored output tree. If omitted, a sibling "
            "directory named '<input>_keyFrames' is created next to the input."
        ),
    )
    parser.add_argument(
        "--frames-dirname",
        default="frames",
        help="Directory name used when discovering frame folders. Default: frames.",
    )
    output_behavior = parser.add_mutually_exclusive_group()
    output_behavior.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    output_behavior.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sequences whose output directory already exists.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        results = extract_keyframes_from_root(
            input_path=args.input_path,
            output_root=args.output_root,
            limit=args.limit,
            baseline_frames=args.baseline_frames,
            smoothing_window=args.smoothing_window,
            workers=args.workers,
            backend=args.backend,
            frames_dirname=args.frames_dirname,
            overwrite=args.overwrite,
            skip_existing=args.skip_existing,
        )
    except RuntimeError as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")

    for result in results:
        if result.skipped:
            print(
                f"{result.frames_dir} -> {result.output_dir} "
                f"(skipped, {result.selected_count} existing keyframes)"
            )
            continue

        print(f"{result.frames_dir} -> {result.output_dir} ({result.selected_count} keyframes)")

    return 0
