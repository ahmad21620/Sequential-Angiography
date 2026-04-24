from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stenosis_detection import (
    DISTINCT_VIEW_ANGLE_DISTANCE_DEGREES,
    DUPLICATE_VIEW_ANGLE_DISTANCE_DEGREES,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    SUPPORT_SCORE_SCALE,
    MultiViewCaseResult,
    MultiViewFusionConfig,
    MultiViewLoadError,
    load_multiview_case,
    run_multiview_fusion,
    save_multiview_case_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run case-level multi-view fusion on temporal fusion outputs.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--case-root",
        help="Directory containing 'views.json' for one case.",
    )
    input_group.add_argument(
        "--input-json",
        help="Direct path to one case-level 'views.json' file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output JSON file for the fused case-level result.",
    )
    parser.add_argument(
        "--duplicate-angle-distance",
        type=float,
        default=DUPLICATE_VIEW_ANGLE_DISTANCE_DEGREES,
        help="Angle distance threshold, in degrees, below which two views are treated as duplicate evidence.",
    )
    parser.add_argument(
        "--distinct-angle-distance",
        type=float,
        default=DISTINCT_VIEW_ANGLE_DISTANCE_DEGREES,
        help="Angle distance threshold, in degrees, at which support is treated as fully distinct.",
    )
    parser.add_argument(
        "--support-score-scale",
        type=float,
        default=SUPPORT_SCORE_SCALE,
        help="Scale factor applied to cross-view support from other views.",
    )
    parser.add_argument(
        "--medium-confidence-threshold",
        type=float,
        default=MEDIUM_CONFIDENCE_THRESHOLD,
        help="Minimum confidence score labeled as medium.",
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=HIGH_CONFIDENCE_THRESHOLD,
        help="Minimum confidence score labeled as high.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.duplicate_angle_distance <= 0.0:
        parser.error("--duplicate-angle-distance must be positive.")
    if args.distinct_angle_distance <= args.duplicate_angle_distance:
        parser.error("--distinct-angle-distance must be greater than --duplicate-angle-distance.")
    if args.support_score_scale < 0.0:
        parser.error("--support-score-scale must be >= 0.0.")
    if not 0.0 <= args.medium_confidence_threshold <= 1.0:
        parser.error("--medium-confidence-threshold must be in the range [0.0, 1.0].")
    if not 0.0 <= args.high_confidence_threshold <= 1.0:
        parser.error("--high-confidence-threshold must be in the range [0.0, 1.0].")
    if args.medium_confidence_threshold > args.high_confidence_threshold:
        parser.error("--medium-confidence-threshold must be <= --high-confidence-threshold.")

    try:
        case_input_path = _resolve_case_input_path(args)
        multiview_case = load_multiview_case(case_input_path)
        print(f"Loaded case '{multiview_case.case_id}' with {multiview_case.view_count} views.")

        case_result = run_multiview_fusion(
            multiview_case,
            config=MultiViewFusionConfig(
                duplicate_view_angle_distance_degrees=args.duplicate_angle_distance,
                distinct_view_angle_distance_degrees=args.distinct_angle_distance,
                support_score_scale=args.support_score_scale,
                medium_confidence_threshold=args.medium_confidence_threshold,
                high_confidence_threshold=args.high_confidence_threshold,
            ),
        )
        _print_fusion_summary(case_result)

        output_path = save_multiview_case_result(case_result, Path(args.output))
        print(f"Saved case-level result: {output_path}")
        return 0
    except (FileNotFoundError, NotADirectoryError, MultiViewLoadError, ValueError) as exc:
        print(f"Multi-view fusion failed: {exc}", file=sys.stderr)
        return 1


def _resolve_case_input_path(args: argparse.Namespace) -> Path:
    if args.case_root is not None:
        case_root = Path(args.case_root)
        if not case_root.exists():
            raise FileNotFoundError(f"Case root does not exist: {case_root}")
        if not case_root.is_dir():
            raise NotADirectoryError(f"Case root is not a directory: {case_root}")
        return case_root / "views.json"

    return Path(args.input_json)


def _print_fusion_summary(case_result: MultiViewCaseResult) -> None:
    final_case_lesion = case_result.final_case_lesion
    supporting_views_text = ", ".join(case_result.supporting_views) if case_result.supporting_views else "none"

    print(f"Lesion candidates: {case_result.fusion_metadata.total_candidate_count}")
    if final_case_lesion is None:
        print("Final severity: none")
        print("Final degree: n/a")
    else:
        print(f"Final severity: {final_case_lesion.severity}")
        print(f"Final degree: {final_case_lesion.median_degree:.3f}")

    print(
        f"Confidence: {case_result.confidence.label} "
        f"({case_result.confidence.score:.3f})"
    )
    print(f"Supporting views: {supporting_views_text}")


if __name__ == "__main__":
    raise SystemExit(main())
