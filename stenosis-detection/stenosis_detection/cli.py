from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from .batch import discover_tree_jobs, process_tree
from .pipeline import PipelineConfig, run_stenosis_detection, run_stenosis_detection_variants
from .visualization import save_detection_outputs
from .yolo.inference import YoloInferenceConfig, discover_yolo_tree_jobs, process_yolo_tree


DETECTOR_CHOICES = ("vessel", "yolo")
VESSEL_ONLY_OPTIONS = {
    "--min-component-area",
    "--remove-border-artifacts",
    "--no-remove-border-artifacts",
    "--border-margin-px",
    "--border-artifact-max-height",
    "--border-artifact-min-width-ratio",
    "--radius-search-range",
    "--radius-vessel-threshold",
    "--radius-outside-fraction-threshold",
    "--radius-min-outside-samples",
    "--segmentation-distance-threshold",
    "--stenosis-threshold",
    "--stenosis-thresholds",
    "--average-radius-threshold",
    "--average-radius-thresholds",
    "--final-point-distance-threshold",
    "--branch-point-exclusion-distance",
}


def build_parser() -> argparse.ArgumentParser:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description="Run stenosis detection on a single slice or an entire mirrored data tree.")

    parser.add_argument(
        "--detector",
        choices=DETECTOR_CHOICES,
        default="vessel",
        help="Frame-level detector to run. Default: vessel.",
    )
    parser.add_argument("--image", help="Path to one original angiography image.")
    parser.add_argument("--mask", help="Path to one binary vessel mask image.")
    parser.add_argument("--output-dir", help="Directory where single-image outputs will be written.")

    parser.add_argument("--images-root", help="Root directory containing original images in a nested tree.")
    parser.add_argument("--masks-root", help="Root directory containing mask images in a mirrored nested tree.")
    parser.add_argument("--output-root", help="Root directory where mirrored batch outputs will be written.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel worker processes for batch mode. Use 0 for all CPU cores; use 1 for serial processing.",
    )

    parser.add_argument("--show", action="store_true", help="Display generated figures after saving them. Single-image mode only.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run items even when the expected output files already exist. Batch mode skips completed slices by default.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip completed batch outputs. This is the default behavior.",
    )
    parser.add_argument(
        "--no-debug-images",
        action="store_true",
        help="Write only '*_stenosis_results.json' outputs and skip PNG debug images.",
    )
    parser.add_argument(
        "--save-review-images",
        action="store_true",
        help="For --detector yolo, also write bbox review PNGs next to JSON outputs.",
    )

    parser.add_argument("--yolo-weights", help="Path to YOLOv8 weights used when --detector yolo.")
    parser.add_argument("--yolo-imgsz", type=int, default=1024, help="YOLO inference image size.")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--yolo-iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--device", help="Device string passed to Ultralytics when --detector yolo.")

    parser.add_argument("--mask-threshold", type=int, default=defaults.mask_threshold, help="Threshold used to binarize vessel masks.")
    parser.add_argument("--min-component-area", type=int, default=defaults.min_component_area, help="Minimum connected-component area retained in vessel masks.")
    parser.add_argument(
        "--remove-border-artifacts",
        action=argparse.BooleanOptionalAction,
        default=defaults.remove_border_artifacts,
        help="Remove long, thin connected components touching mask borders. Use --no-remove-border-artifacts to disable.",
    )
    parser.add_argument("--border-margin-px", type=int, default=defaults.border_margin_px, help="Border margin used when detecting mask artifacts.")
    parser.add_argument(
        "--border-artifact-max-height",
        type=int,
        default=defaults.border_artifact_max_height,
        help="Maximum component height for long, thin border artifact removal.",
    )
    parser.add_argument(
        "--border-artifact-min-width-ratio",
        type=float,
        default=defaults.border_artifact_min_width_ratio,
        help="Minimum component width as a fraction of image width for border artifact removal.",
    )
    parser.add_argument("--radius-search-range", type=float, default=defaults.radius_search_range, help="Radius search range used by the radius estimator.")
    parser.add_argument(
        "--radius-vessel-threshold",
        type=int,
        default=defaults.radius_vessel_threshold,
        help="Pixel threshold used to treat sampled radius points as vessel.",
    )
    parser.add_argument(
        "--radius-outside-fraction-threshold",
        type=float,
        default=defaults.radius_outside_fraction_threshold,
        help="Minimum outside-pixel fraction needed to stop radius search.",
    )
    parser.add_argument(
        "--radius-min-outside-samples",
        type=int,
        default=defaults.radius_min_outside_samples,
        help="Minimum outside-pixel sample count needed to stop radius search.",
    )
    parser.add_argument("--segmentation-distance-threshold", type=float, default=defaults.segmentation_distance_threshold, help="Distance threshold for filtering nearby segmentation points.")
    parser.add_argument("--stenosis-threshold", type=float, default=defaults.stenosis_threshold, help="Threshold used for stenosis degree filtering.")
    parser.add_argument("--stenosis-thresholds", help="Comma-separated stenosis thresholds to evaluate in one shared run, e.g. 0.20,0.25,0.30.")
    parser.add_argument("--average-radius-threshold", type=float, default=defaults.average_radius_threshold, help="Average path radius threshold used for stenosis filtering.")
    parser.add_argument("--average-radius-thresholds", help="Comma-separated average-radius thresholds to evaluate in one shared run, e.g. 3.0,4.0,5.0.")
    parser.add_argument("--final-point-distance-threshold", type=float, default=defaults.final_point_distance_threshold, help="Distance threshold used during final stenosis point filtering.")
    parser.add_argument("--branch-point-exclusion-distance", type=float, default=defaults.branch_point_exclusion_distance, help="Drop stenosis candidates within this distance of detected branch points. Use 0 to disable.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_args)
    if args.workers < 0:
        parser.error("--workers must be 0 or greater.")
    if args.show and args.no_debug_images:
        parser.error("--show cannot be used with --no-debug-images.")
    if args.overwrite and args.skip_existing:
        parser.error("--overwrite and --skip-existing are mutually exclusive.")
    if args.save_review_images and args.no_debug_images:
        parser.error("--save-review-images cannot be used with --no-debug-images.")
    _validate_yolo_options(parser, args)

    single_mode = args.image or args.mask or args.output_dir
    batch_mode = args.images_root or args.masks_root or args.output_root

    if single_mode and batch_mode:
        parser.error("Use either single-image arguments (--image, --mask, --output-dir) or tree arguments (--images-root, --masks-root, --output-root), not both.")
    if not single_mode and not batch_mode:
        parser.error("Provide either single-image arguments or tree arguments.")

    if args.detector == "yolo":
        if single_mode:
            parser.error("--detector yolo currently supports tree mode with --images-root and --output-root.")
        _validate_yolo_batch_mode(parser, args)
        _warn_ignored_options(
            raw_args,
            VESSEL_ONLY_OPTIONS,
            "--detector yolo ignores vessel-mask/radius-specific options",
        )
        return _run_yolo_batch(args)

    config = _build_pipeline_config(args)
    write_debug_images = not args.no_debug_images
    try:
        threshold_variants = _build_threshold_variants(args, config)
    except ValueError as exc:
        parser.error(str(exc))

    if single_mode:
        _validate_single_mode(parser, args)
        if threshold_variants is None:
            result = run_stenosis_detection(
                image_path=Path(args.image),
                mask_path=Path(args.mask),
                config=config,
            )
            output_files = save_detection_outputs(
                result,
                Path(args.output_dir),
                show=args.show,
                write_debug_images=write_debug_images,
            )

            print("Stenosis detection completed.")
            print(f"Detected stenosis points: {len(result.stenosis_points_xy)}")
            for name, path in output_files.items():
                print(f"{name}: {path}")
        else:
            results = run_stenosis_detection_variants(
                image_path=Path(args.image),
                mask_path=Path(args.mask),
                configs=[variant_config for _, variant_config in threshold_variants],
            )
            print("Stenosis detection completed.")
            for (variant_name, _), result in zip(threshold_variants, results, strict=True):
                output_files = save_detection_outputs(
                    result,
                    Path(args.output_dir) / variant_name,
                    show=args.show,
                    write_debug_images=write_debug_images,
                )
                print(f"{variant_name}: {len(result.stenosis_points_xy)} stenosis points")
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
        workers=args.workers,
        threshold_variants=threshold_variants,
        write_debug_images=write_debug_images,
    )

    print("Batch stenosis detection completed.")
    print(f"Total slices discovered: {summary.total_jobs}")
    print(f"Workers: {summary.workers}")
    if threshold_variants is not None:
        print(f"Threshold variants: {len(threshold_variants)}")
    print(f"Debug images: {'yes' if write_debug_images else 'no'}")
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


def _validate_yolo_batch_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    missing = [name for name in ("images_root", "output_root", "yolo_weights") if getattr(args, name) is None]
    if missing:
        parser.error(
            "YOLO tree mode requires --images-root, --output-root, and --yolo-weights. "
            f"Missing: {', '.join('--' + item.replace('_', '-') for item in missing)}"
        )
    if not Path(args.yolo_weights).is_file():
        parser.error(f"--yolo-weights is not a file: {args.yolo_weights}")
    if args.show:
        parser.error("--show is only supported in single-image vessel mode.")


def _validate_yolo_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.yolo_imgsz is not None and args.yolo_imgsz < 1:
        parser.error("--yolo-imgsz must be at least 1.")
    if not 0.0 <= args.yolo_conf <= 1.0:
        parser.error("--yolo-conf must be in the range [0.0, 1.0].")
    if not 0.0 <= args.yolo_iou <= 1.0:
        parser.error("--yolo-iou must be in the range [0.0, 1.0].")


def _run_yolo_batch(args: argparse.Namespace) -> int:
    jobs = discover_yolo_tree_jobs(args.images_root, masks_root=args.masks_root)
    summary = process_yolo_tree(
        jobs,
        output_root=args.output_root,
        images_root=args.images_root,
        masks_root=args.masks_root,
        config=YoloInferenceConfig(
            weights=args.yolo_weights,
            imgsz=args.yolo_imgsz,
            conf=args.yolo_conf,
            iou=args.yolo_iou,
            device=args.device,
            mask_threshold=args.mask_threshold,
        ),
        skip_existing=not args.overwrite,
        workers=args.workers,
        write_review_images=args.save_review_images,
    )

    print("YOLO stenosis detection completed.")
    print(f"Total frames discovered: {summary.total_jobs}")
    print(f"Workers: {summary.workers}")
    print(f"Review images: {'yes' if args.save_review_images else 'no'}")
    print(f"Processed: {summary.processed}")
    print(f"Skipped existing: {summary.skipped_existing}")
    print(f"Failed: {summary.failed}")
    print(f"Summary JSON: {Path(args.output_root) / 'batch_summary.json'}")
    return 0 if summary.failed == 0 else 1


def _build_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        mask_threshold=args.mask_threshold,
        min_component_area=args.min_component_area,
        remove_border_artifacts=args.remove_border_artifacts,
        border_margin_px=args.border_margin_px,
        border_artifact_max_height=args.border_artifact_max_height,
        border_artifact_min_width_ratio=args.border_artifact_min_width_ratio,
        radius_search_range=args.radius_search_range,
        radius_vessel_threshold=args.radius_vessel_threshold,
        radius_outside_fraction_threshold=args.radius_outside_fraction_threshold,
        radius_min_outside_samples=args.radius_min_outside_samples,
        segmentation_distance_threshold=args.segmentation_distance_threshold,
        stenosis_threshold=args.stenosis_threshold,
        average_radius_threshold=args.average_radius_threshold,
        final_point_distance_threshold=args.final_point_distance_threshold,
        branch_point_exclusion_distance=args.branch_point_exclusion_distance,
    )


def _build_threshold_variants(
    args: argparse.Namespace,
    config: PipelineConfig,
) -> list[tuple[str, PipelineConfig]] | None:
    if args.stenosis_thresholds is None and args.average_radius_thresholds is None:
        return None

    stenosis_thresholds = _parse_float_list(args.stenosis_thresholds, "--stenosis-thresholds")
    average_radius_thresholds = _parse_float_list(args.average_radius_thresholds, "--average-radius-thresholds")
    if not stenosis_thresholds:
        stenosis_thresholds = [config.stenosis_threshold]
    if not average_radius_thresholds:
        average_radius_thresholds = [config.average_radius_threshold]

    variants: list[tuple[str, PipelineConfig]] = []
    for stenosis_threshold in stenosis_thresholds:
        for average_radius_threshold in average_radius_thresholds:
            variant_name = _threshold_variant_name(stenosis_threshold, average_radius_threshold)
            variant_config = replace(
                config,
                stenosis_threshold=stenosis_threshold,
                average_radius_threshold=average_radius_threshold,
            )
            variants.append((variant_name, variant_config))
    return variants


def _parse_float_list(raw_value: str | None, option_name: str) -> list[float]:
    if raw_value is None:
        return []

    values: list[float] = []
    for item in raw_value.split(","):
        clean_item = item.strip()
        if not clean_item:
            continue
        try:
            values.append(float(clean_item))
        except ValueError as exc:
            raise ValueError(f"{option_name} must contain comma-separated numbers.") from exc
    if not values:
        raise ValueError(f"{option_name} must contain at least one number.")
    return values


def _threshold_variant_name(stenosis_threshold: float, average_radius_threshold: float) -> str:
    return (
        f"stenosis_threshold_{_format_threshold_value(stenosis_threshold)}"
        f"__average_radius_threshold_{_format_threshold_value(average_radius_threshold)}"
    )


def _format_threshold_value(value: float) -> str:
    return f"{value:g}".replace("-", "minus_").replace(".", "p")


def _warn_ignored_options(raw_args: list[str], option_names: set[str], message: str) -> None:
    used_options = sorted(option for option in option_names if _option_used(raw_args, option))
    if not used_options:
        return
    print(f"Warning: {message}: {', '.join(used_options)}.", file=sys.stderr)


def _option_used(raw_args: list[str], option: str) -> bool:
    option_prefix = f"{option}="
    return any(argument == option or argument.startswith(option_prefix) for argument in raw_args)
