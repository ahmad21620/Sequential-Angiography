from __future__ import annotations

import argparse
from pathlib import Path
import sys

from stenosis_detection import DEFAULT_VIEW_FRAME_COUNT, MultiViewFusionConfig, PipelineConfig, TemporalLoadError, VIDEO_FORMATS
from stenosis_detection.parameter_sweep import run_parameter_sweep
from stenosis_detection.temporal import DEFAULT_VIDEO_FPS


def build_parser() -> argparse.ArgumentParser:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(
        description="Run an optimized frame-level and temporal-fusion parameter sweep.",
    )
    parser.add_argument(
        "--frame-detector",
        choices=["vessel", "yolo"],
        default="vessel",
        help="Frame-level detector used to produce sweep frame results. Default: vessel.",
    )
    parser.add_argument("--images-root", required=True, help="Root containing extracted keyframe images.")
    parser.add_argument(
        "--masks-root",
        help="Root containing mirrored vessel masks. Required for --frame-detector vessel; optional skeleton support for YOLO.",
    )
    parser.add_argument("--output-root", required=True, help="Root where sweep outputs will be written.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Frame-level worker processes. Use 0 for all CPU cores. Default: 1.",
    )
    parser.add_argument(
        "--temporal-workers",
        type=int,
        default=1,
        help="Temporal-fusion worker processes. Use 0 for all CPU cores. Default: 1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run variants even when their expected outputs already exist.",
    )
    parser.add_argument(
        "--no-debug-images",
        action="store_true",
        help="Write only frame-level JSON outputs and skip frame-level PNG debug images.",
    )
    parser.add_argument(
        "--no-temporal-images",
        action="store_true",
        help="Write temporal JSON outputs but skip temporal summary PNG images.",
    )
    parser.add_argument(
        "--allow-variable-frame-count",
        action="store_true",
        help="Allow each view to have its own frame count. Use this for CADICA.",
    )
    parser.add_argument(
        "--expected-frame-count",
        type=int,
        default=DEFAULT_VIEW_FRAME_COUNT,
        help="Expected frame-result count per view when variable counts are not allowed.",
    )
    parser.add_argument("--write-video", action="store_true", help="Write temporal demo videos for each temporal variant.")
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS, help="FPS for optional temporal videos.")
    parser.add_argument("--video-format", choices=VIDEO_FORMATS, default="mp4", help="Optional temporal video format.")
    parser.add_argument(
        "--run-multiview",
        action="store_true",
        help="After each temporal variant, run one fixed multi-view fusion pass.",
    )
    parser.add_argument(
        "--multiview-case-root-tree",
        help="Root containing case views.json files. Defaults to --images-root.",
    )
    parser.add_argument(
        "--multiview-output-root",
        help="Root where multi-view sweep outputs will be written. Defaults to <output-root>/multiview_results.",
    )
    parser.add_argument(
        "--multiview-view-diversity-mode",
        choices=["angle", "projection_group", "auto"],
        default="angle",
        help="Multi-view diversity mode. Use projection_group for CADICA.",
    )
    parser.add_argument(
        "--multiview-split-by-coronary-side",
        action="store_true",
        help="Run separate left/right multi-view fusion using coronary_side metadata.",
    )

    parser.add_argument("--stenosis-thresholds", help="Comma-separated frame-level stenosis thresholds.")
    parser.add_argument("--average-radius-thresholds", help="Comma-separated average-radius thresholds.")
    parser.add_argument("--radius-outside-fraction-thresholds", help="Comma-separated radius outside-fraction thresholds.")
    parser.add_argument("--radius-min-outside-samples-values", help="Comma-separated radius minimum outside-sample counts.")
    parser.add_argument("--yolo-weights", help="YOLOv8 weights used when --frame-detector yolo.")
    parser.add_argument("--yolo-conf-thresholds", help="Comma-separated YOLO confidence thresholds.")
    parser.add_argument("--yolo-iou-thresholds", help="Comma-separated YOLO NMS IoU thresholds.")
    parser.add_argument("--yolo-imgsz-values", help="Comma-separated YOLO inference image sizes.")
    parser.add_argument("--device", help="Device string passed to Ultralytics for YOLO frame sweeps.")
    parser.add_argument(
        "--save-review-images",
        action="store_true",
        help="For YOLO frame sweeps, also write bbox review PNGs next to JSON outputs.",
    )
    parser.add_argument("--min-supporting-frames-values", help="Comma-separated temporal minimum supporting-frame counts.")
    parser.add_argument("--min-persistence-ratios", help="Comma-separated temporal minimum persistence ratios.")

    parser.add_argument("--mask-threshold", type=int, default=defaults.mask_threshold)
    parser.add_argument("--min-component-area", type=int, default=defaults.min_component_area)
    parser.add_argument(
        "--remove-border-artifacts",
        action=argparse.BooleanOptionalAction,
        default=defaults.remove_border_artifacts,
    )
    parser.add_argument("--border-margin-px", type=int, default=defaults.border_margin_px)
    parser.add_argument("--border-artifact-max-height", type=int, default=defaults.border_artifact_max_height)
    parser.add_argument("--border-artifact-min-width-ratio", type=float, default=defaults.border_artifact_min_width_ratio)
    parser.add_argument("--radius-search-range", type=float, default=defaults.radius_search_range)
    parser.add_argument("--radius-vessel-threshold", type=int, default=defaults.radius_vessel_threshold)
    parser.add_argument("--segmentation-distance-threshold", type=float, default=defaults.segmentation_distance_threshold)
    parser.add_argument("--final-point-distance-threshold", type=float, default=defaults.final_point_distance_threshold)
    parser.add_argument("--branch-point-exclusion-distance", type=float, default=defaults.branch_point_exclusion_distance)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_args)
    if args.workers < 0:
        parser.error("--workers must be 0 or greater.")
    if args.temporal_workers < 0:
        parser.error("--temporal-workers must be 0 or greater.")
    if not args.allow_variable_frame_count and args.expected_frame_count < 1:
        parser.error("--expected-frame-count must be at least 1.")
    if args.video_fps <= 0.0:
        parser.error("--video-fps must be positive.")
    if args.save_review_images and args.no_debug_images:
        parser.error("--save-review-images cannot be used with --no-debug-images.")
    if args.frame_detector == "vessel" and args.masks_root is None:
        parser.error("--masks-root is required when --frame-detector vessel.")
    if args.frame_detector == "yolo" and args.yolo_weights is None:
        parser.error("--yolo-weights is required when --frame-detector yolo.")
    if args.frame_detector == "yolo" and args.yolo_weights is not None and not Path(args.yolo_weights).is_file():
        parser.error(f"--yolo-weights is not a file: {args.yolo_weights}")
    if args.frame_detector == "yolo":
        _warn_ignored_options(
            raw_args,
            {
                "--stenosis-thresholds",
                "--average-radius-thresholds",
                "--radius-outside-fraction-thresholds",
                "--radius-min-outside-samples-values",
                "--radius-search-range",
                "--radius-vessel-threshold",
                "--segmentation-distance-threshold",
                "--final-point-distance-threshold",
                "--branch-point-exclusion-distance",
                "--min-component-area",
                "--remove-border-artifacts",
                "--no-remove-border-artifacts",
                "--border-margin-px",
                "--border-artifact-max-height",
                "--border-artifact-min-width-ratio",
            },
            "--frame-detector yolo ignores vessel-mask/radius-specific options",
        )

    try:
        yolo_conf_thresholds = (
            _parse_float_list(args.yolo_conf_thresholds, "--yolo-conf-thresholds")
            if args.frame_detector == "yolo"
            else None
        )
        yolo_iou_thresholds = (
            _parse_float_list(args.yolo_iou_thresholds, "--yolo-iou-thresholds")
            if args.frame_detector == "yolo"
            else None
        )
        yolo_imgsz_values = (
            _parse_int_list(args.yolo_imgsz_values, "--yolo-imgsz-values")
            if args.frame_detector == "yolo"
            else None
        )
        result = run_parameter_sweep(
            images_root=args.images_root,
            masks_root=args.masks_root,
            output_root=args.output_root,
            frame_detector=args.frame_detector,
            stenosis_thresholds=(
                _parse_float_list(args.stenosis_thresholds, "--stenosis-thresholds")
                if args.frame_detector == "vessel"
                else None
            ),
            average_radius_thresholds=(
                _parse_float_list(args.average_radius_thresholds, "--average-radius-thresholds")
                if args.frame_detector == "vessel"
                else None
            ),
            radius_outside_fraction_thresholds=_parse_float_list(
                args.radius_outside_fraction_thresholds,
                "--radius-outside-fraction-thresholds",
            )
            if args.frame_detector == "vessel"
            else None,
            radius_min_outside_samples_values=(
                _parse_int_list(
                    args.radius_min_outside_samples_values,
                    "--radius-min-outside-samples-values",
                )
                if args.frame_detector == "vessel"
                else None
            ),
            yolo_weights=args.yolo_weights,
            yolo_conf_thresholds=yolo_conf_thresholds,
            yolo_iou_thresholds=yolo_iou_thresholds,
            yolo_imgsz_values=yolo_imgsz_values,
            yolo_device=args.device,
            min_supporting_frames_values=_parse_int_list(
                args.min_supporting_frames_values,
                "--min-supporting-frames-values",
            ),
            min_persistence_ratios=_parse_float_list(args.min_persistence_ratios, "--min-persistence-ratios"),
            base_config=_build_base_config(args),
            workers=args.workers,
            temporal_workers=args.temporal_workers,
            allow_variable_frame_count=args.allow_variable_frame_count,
            expected_frame_count=args.expected_frame_count,
            skip_existing=not args.overwrite,
            write_debug_images=not args.no_debug_images,
            write_temporal_images=not args.no_temporal_images,
            write_video=args.write_video,
            video_fps=args.video_fps,
            video_format=args.video_format,
            run_multiview=args.run_multiview,
            multiview_case_root_tree=args.multiview_case_root_tree,
            multiview_output_root=args.multiview_output_root,
            multiview_config=MultiViewFusionConfig(view_diversity_mode=args.multiview_view_diversity_mode),
            split_multiview_by_coronary_side=args.multiview_split_by_coronary_side,
            write_yolo_review_images=args.save_review_images,
        )
    except (FileNotFoundError, NotADirectoryError, TemporalLoadError, ValueError, OSError) as exc:
        print(f"Parameter sweep failed: {exc}", file=sys.stderr)
        return 2

    print(f"Frame variants: {len(result.frame_variants)}")
    print(f"Temporal variants: {len(result.temporal_variants)}")
    print(f"Frame results root: {result.frame_results_root}")
    print(f"Temporal results root: {result.temporal_results_root}")
    if result.multiview_results_root is not None:
        print(f"Multi-view results root: {result.multiview_results_root}")
    print(f"Frame failures: {result.frame_summary.failed}")
    print(f"Temporal failed jobs: {result.temporal_summary.failed_jobs}")
    multiview_failed_cases = 0 if result.multiview_summary is None else result.multiview_summary.failed_cases
    if result.multiview_summary is not None:
        print(f"Multi-view failed cases: {multiview_failed_cases}")
    print(f"Summary: {result.summary_json}")
    return (
        0
        if result.frame_summary.failed == 0
        and result.temporal_summary.failed_jobs == 0
        and multiview_failed_cases == 0
        else 1
    )


def _build_base_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        mask_threshold=args.mask_threshold,
        min_component_area=args.min_component_area,
        remove_border_artifacts=args.remove_border_artifacts,
        border_margin_px=args.border_margin_px,
        border_artifact_max_height=args.border_artifact_max_height,
        border_artifact_min_width_ratio=args.border_artifact_min_width_ratio,
        radius_search_range=args.radius_search_range,
        radius_vessel_threshold=args.radius_vessel_threshold,
        segmentation_distance_threshold=args.segmentation_distance_threshold,
        final_point_distance_threshold=args.final_point_distance_threshold,
        branch_point_exclusion_distance=args.branch_point_exclusion_distance,
    )


def _parse_float_list(raw_value: str | None, option_name: str) -> list[float] | None:
    return _parse_list(raw_value, option_name, float)


def _parse_int_list(raw_value: str | None, option_name: str) -> list[int] | None:
    return _parse_list(raw_value, option_name, int)


def _parse_list(raw_value: str | None, option_name: str, parser) -> list | None:
    if raw_value is None:
        return None

    values = []
    for item in raw_value.split(","):
        clean_item = item.strip()
        if not clean_item:
            continue
        try:
            values.append(parser(clean_item))
        except ValueError as exc:
            raise ValueError(f"{option_name} must contain comma-separated values.") from exc
    if not values:
        raise ValueError(f"{option_name} must contain at least one value.")
    return values


def _warn_ignored_options(raw_args: list[str], option_names: set[str], message: str) -> None:
    used_options = sorted(option for option in option_names if _option_used(raw_args, option))
    if not used_options:
        return
    print(f"Warning: {message}: {', '.join(used_options)}.", file=sys.stderr)


def _option_used(raw_args: list[str], option: str) -> bool:
    option_prefix = f"{option}="
    return any(argument == option or argument.startswith(option_prefix) for argument in raw_args)


if __name__ == "__main__":
    raise SystemExit(main())
