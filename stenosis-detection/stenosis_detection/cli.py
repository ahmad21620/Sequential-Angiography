from __future__ import annotations

import argparse
from pathlib import Path

from .batch import discover_tree_jobs, process_tree
from .pipeline import PipelineConfig, run_stenosis_detection
from .visualization import save_detection_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run stenosis detection on a single slice or an entire mirrored data tree.")

    parser.add_argument("--image", help="Path to one original angiography image.")
    parser.add_argument("--mask", help="Path to one binary vessel mask image.")
    parser.add_argument("--output-dir", help="Directory where single-image outputs will be written.")

    parser.add_argument("--images-root", help="Root directory containing original images in a nested tree.")
    parser.add_argument("--masks-root", help="Root directory containing mask images in a mirrored nested tree.")
    parser.add_argument("--output-root", help="Root directory where mirrored batch outputs will be written.")

    parser.add_argument("--show", action="store_true", help="Display generated figures after saving them. Single-image mode only.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run items even when the expected output files already exist. Batch mode skips completed slices by default.",
    )
    parser.add_argument("--radius-search-range", type=float, default=110.0, help="Radius search range used by the radius estimator.")
    parser.add_argument("--segmentation-distance-threshold", type=float, default=8.0, help="Distance threshold for filtering nearby segmentation points.")
    parser.add_argument("--stenosis-threshold", type=float, default=0.25, help="Threshold used for stenosis degree filtering.")
    parser.add_argument("--average-radius-threshold", type=float, default=4.0, help="Average path radius threshold used for stenosis filtering.")
    parser.add_argument("--final-point-distance-threshold", type=float, default=10.0, help="Distance threshold used during final stenosis point filtering.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = PipelineConfig(
        radius_search_range=args.radius_search_range,
        segmentation_distance_threshold=args.segmentation_distance_threshold,
        stenosis_threshold=args.stenosis_threshold,
        average_radius_threshold=args.average_radius_threshold,
        final_point_distance_threshold=args.final_point_distance_threshold,
    )

    single_mode = args.image or args.mask or args.output_dir
    batch_mode = args.images_root or args.masks_root or args.output_root

    if single_mode and batch_mode:
        parser.error("Use either single-image arguments (--image, --mask, --output-dir) or tree arguments (--images-root, --masks-root, --output-root), not both.")
    if not single_mode and not batch_mode:
        parser.error("Provide either single-image arguments or tree arguments.")

    if single_mode:
        _validate_single_mode(parser, args)
        result = run_stenosis_detection(
            image_path=Path(args.image),
            mask_path=Path(args.mask),
            config=config,
        )
        output_files = save_detection_outputs(result, Path(args.output_dir), show=args.show)

        print("Stenosis detection completed.")
        print(f"Detected stenosis points: {len(result.stenosis_points_xy)}")
        for name, path in output_files.items():
            print(f"{name}: {path}")
        return 0

    _validate_batch_mode(parser, args)
    jobs = discover_tree_jobs(args.images_root, args.masks_root)
    summary = process_tree(
        jobs,
        args.output_root,
        images_root=args.images_root,
        masks_root=args.masks_root,
        config=config,
        skip_existing=not args.overwrite,
    )

    print("Batch stenosis detection completed.")
    print(f"Total slices discovered: {summary.total_jobs}")
    print(f"Processed: {summary.processed}")
    print(f"Skipped existing: {summary.skipped_existing}")
    print(f"Failed: {summary.failed}")
    print(f"Summary JSON: {Path(args.output_root) / 'batch_summary.json'}")
    return 0 if summary.failed == 0 else 1


def _validate_single_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    missing = [name for name in ("image", "mask", "output_dir") if getattr(args, name) is None]
    if missing:
        parser.error(f"Single-image mode requires --image, --mask, and --output-dir. Missing: {', '.join('--' + item.replace('_', '-') for item in missing)}")


def _validate_batch_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    missing = [name for name in ("images_root", "masks_root", "output_root") if getattr(args, name) is None]
    if missing:
        parser.error(f"Tree mode requires --images-root, --masks-root, and --output-root. Missing: {', '.join('--' + item.replace('_', '-') for item in missing)}")
    if args.show:
        parser.error("--show is only supported in single-image mode.")
