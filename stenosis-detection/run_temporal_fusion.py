from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import sys
from pathlib import Path

from stenosis_detection import (
    DEFAULT_MIN_PERSISTENCE_RATIO,
    DEFAULT_MIN_SUPPORTING_FRAMES,
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIEW_FRAME_COUNT,
    VIDEO_FORMATS,
    TemporalLoadError,
    ViewSequence,
    ViewLevelResult,
    build_view_video_path,
    build_view_visualization_paths,
    load_view_sequences,
    load_view_sequence,
    run_temporal_fusion_on_view_sequence,
    save_view_demo_video,
    save_view_level_result,
    save_view_visualization_outputs,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is absent.
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, *, total=None, desc=None, unit=None, dynamic_ncols=None):
            self.iterable = iterable
            self.total = total
            self.desc = desc or "Progress"
            self.unit = unit or "item"
            self.count = 0
            print(f"{self.desc}: 0/{self.total if self.total is not None else '?'} {self.unit}")

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            for item in self.iterable:
                yield item
                self.update(1)

        def update(self, n=1):
            self.count += n

        def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
            return None

        def write(self, message):
            print(message)

        def close(self):
            print(f"{self.desc}: {self.count}/{self.total if self.total is not None else '?'} {self.unit}")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()


@dataclass(frozen=True, slots=True)
class FrameCountSkip:
    view_id: str
    frame_count: int
    expected_frame_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run temporal fusion on one or more views of frame-level stenosis results.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--results-root",
        help="Directory containing frame-level '*_stenosis_results.json' files for one view, or a larger root that contains multiple views.",
    )
    input_group.add_argument(
        "--frame-results",
        nargs="+",
        help="Explicit list of frame-level '*_stenosis_results.json' files to fuse.",
    )

    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--output",
        help="Path to the output JSON file for one fused view-level result.",
    )
    output_group.add_argument(
        "--output-root",
        help="Root directory under which one or more fused view-level results will be written.",
    )
    parser.add_argument(
        "--view-id",
        help="Exact view_id to fuse when the input contains multiple views. When omitted, all discovered views are processed if --output-root is used.",
    )
    parser.add_argument(
        "--expected-frame-count",
        type=int,
        default=DEFAULT_VIEW_FRAME_COUNT,
        help="Expected number of frame-level results for the selected view.",
    )
    parser.add_argument(
        "--min-supporting-frames",
        type=int,
        default=DEFAULT_MIN_SUPPORTING_FRAMES,
        help="Minimum number of supporting frames required for a persistent lesion.",
    )
    parser.add_argument(
        "--min-persistence-ratio",
        type=float,
        default=DEFAULT_MIN_PERSISTENCE_RATIO,
        help="Minimum supporting-frame ratio required for a persistent lesion.",
    )
    parser.add_argument(
        "--write-video",
        action="store_true",
        help="Write a minimal temporal demonstration video next to the fused JSON output.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=DEFAULT_VIDEO_FPS,
        help="Frames per second used for the optional temporal demo video.",
    )
    parser.add_argument(
        "--video-format",
        choices=VIDEO_FORMATS,
        default="mp4",
        help="Output format for the optional temporal demo video.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a view only when the full expected output set for this run already exists and is non-empty.",
    )
    parser.add_argument(
        "--skip-frame-count-mismatches",
        action="store_true",
        help=(
            "Skip views whose frame-result count does not match --expected-frame-count, "
            "then print a tally at the end instead of failing the whole run."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.expected_frame_count < 1:
        parser.error("--expected-frame-count must be at least 1.")
    if args.min_supporting_frames < 1:
        parser.error("--min-supporting-frames must be at least 1.")
    if not 0.0 < args.min_persistence_ratio <= 1.0:
        parser.error("--min-persistence-ratio must be in the range (0.0, 1.0].")
    if args.video_fps <= 0.0:
        parser.error("--video-fps must be positive.")

    try:
        result_source = _resolve_result_source(args)
        if args.view_id is not None:
            frame_count_skips: list[FrameCountSkip] = []
            if args.skip_frame_count_mismatches:
                view_sequence = load_view_sequence(
                    result_source,
                    expected_frame_count=None,
                    view_id=args.view_id,
                )
                view_sequences, frame_count_skips = _filter_view_sequences_by_expected_frame_count(
                    [view_sequence],
                    expected_frame_count=args.expected_frame_count,
                )
                if not view_sequences:
                    print("Processed 0 views.")
                    _print_frame_count_skip_summary(frame_count_skips)
                    return 0
                view_sequence = view_sequences[0]
            else:
                view_sequence = load_view_sequence(
                    result_source,
                    expected_frame_count=args.expected_frame_count,
                    view_id=args.view_id,
                )
            output_path = _resolve_single_output_path(args, view_sequence, result_source=result_source)
            if args.skip_existing and _is_view_fully_processed(
                output_path,
                write_video=args.write_video,
                video_format=args.video_format,
            ):
                print(f"Skipped view '{view_sequence.view_id}' because all expected outputs already exist: {output_path}")
                if args.skip_frame_count_mismatches:
                    _print_frame_count_skip_summary(frame_count_skips)
                return 0
            _run_single_view(
                view_sequence,
                output_path=output_path,
                min_supporting_frames=args.min_supporting_frames,
                min_persistence_ratio=args.min_persistence_ratio,
                write_video=args.write_video,
                video_fps=args.video_fps,
                video_format=args.video_format,
            )
            if args.skip_frame_count_mismatches:
                _print_frame_count_skip_summary(frame_count_skips)
            return 0

        frame_count_skips = []
        if args.skip_frame_count_mismatches:
            discovered_view_sequences = load_view_sequences(
                result_source,
                expected_frame_count=None,
            )
            view_sequences, frame_count_skips = _filter_view_sequences_by_expected_frame_count(
                discovered_view_sequences,
                expected_frame_count=args.expected_frame_count,
            )
        else:
            view_sequences = load_view_sequences(
                result_source,
                expected_frame_count=args.expected_frame_count,
            )

        if not view_sequences:
            print("Processed 0 views.")
            if args.skip_existing:
                print("Skipped 0 views with complete existing outputs.")
            if args.skip_frame_count_mismatches:
                _print_frame_count_skip_summary(frame_count_skips)
            return 0

        if len(view_sequences) == 1 and not frame_count_skips:
            only_view = view_sequences[0]
            output_path = _resolve_single_output_path(args, only_view, result_source=result_source)
            if args.skip_existing and _is_view_fully_processed(
                output_path,
                write_video=args.write_video,
                video_format=args.video_format,
            ):
                print(f"Skipped view '{only_view.view_id}' because all expected outputs already exist: {output_path}")
                return 0
            _run_single_view(
                only_view,
                output_path=output_path,
                min_supporting_frames=args.min_supporting_frames,
                min_persistence_ratio=args.min_persistence_ratio,
                write_video=args.write_video,
                video_fps=args.video_fps,
                video_format=args.video_format,
            )
            if args.skip_frame_count_mismatches:
                _print_frame_count_skip_summary(frame_count_skips)
            return 0

        if args.output is not None:
            parser.error("--output can only be used when exactly one view is selected. Use --output-root for batch temporal fusion.")

        processed = 0
        skipped = 0
        with tqdm(total=len(view_sequences), desc="Processing views", unit="view", dynamic_ncols=True) as progress:
            for view_sequence in view_sequences:
                output_path = _build_view_output_path(args.output_root, view_sequence, result_source=result_source)
                if args.skip_existing and _is_view_fully_processed(
                    output_path,
                    write_video=args.write_video,
                    video_format=args.video_format,
                ):
                    skipped += 1
                    progress.write(
                        f"Skipped view '{view_sequence.view_id}' because all expected outputs already exist: {output_path}"
                    )
                    progress.update(1)
                    progress.set_postfix(
                        processed=processed,
                        skipped=skipped,
                        frame_count_skipped=len(frame_count_skips),
                    )
                    continue

                _run_single_view(
                    view_sequence,
                    output_path=output_path,
                    min_supporting_frames=args.min_supporting_frames,
                    min_persistence_ratio=args.min_persistence_ratio,
                    write_video=args.write_video,
                    video_fps=args.video_fps,
                    video_format=args.video_format,
                    log=progress.write,
                )
                processed += 1
                progress.update(1)
                progress.set_postfix(
                    processed=processed,
                    skipped=skipped,
                    frame_count_skipped=len(frame_count_skips),
                )

        print(f"Processed {processed} views.")
        if args.skip_existing:
            print(f"Skipped {skipped} views with complete existing outputs.")
        if args.skip_frame_count_mismatches:
            _print_frame_count_skip_summary(frame_count_skips)
        return 0
    except (FileNotFoundError, NotADirectoryError, TemporalLoadError, ValueError) as exc:
        print(f"Temporal fusion failed: {exc}", file=sys.stderr)
        return 1


def _resolve_result_source(args: argparse.Namespace) -> str | list[str]:
    if args.results_root is not None:
        return args.results_root

    return list(args.frame_results)


def _filter_view_sequences_by_expected_frame_count(
    view_sequences: list[ViewSequence],
    *,
    expected_frame_count: int,
) -> tuple[list[ViewSequence], list[FrameCountSkip]]:
    accepted_sequences: list[ViewSequence] = []
    skipped_views: list[FrameCountSkip] = []

    for view_sequence in view_sequences:
        if view_sequence.frame_count == expected_frame_count:
            accepted_sequences.append(view_sequence)
            continue

        skipped_views.append(
            FrameCountSkip(
                view_id=view_sequence.view_id,
                frame_count=view_sequence.frame_count,
                expected_frame_count=expected_frame_count,
            )
        )

    return accepted_sequences, skipped_views


def _print_frame_count_skip_summary(skipped_views: list[FrameCountSkip], *, log=print) -> None:
    if not skipped_views:
        log("Skipped 0 views with frame-count mismatches.")
        return

    expected_counts = sorted({skipped_view.expected_frame_count for skipped_view in skipped_views})
    expected_summary = ", ".join(str(expected_count) for expected_count in expected_counts)
    log(
        f"Skipped {len(skipped_views)} {_pluralize('view', len(skipped_views))} "
        f"with frame-count mismatches (expected {expected_summary} frame results)."
    )

    frame_count_tally = Counter(skipped_view.frame_count for skipped_view in skipped_views)
    for frame_count, view_count in sorted(frame_count_tally.items()):
        log(f"  {frame_count} frame results: {view_count} {_pluralize('view', view_count)}")


def _pluralize(word: str, count: int) -> str:
    if count == 1:
        return word
    return f"{word}s"


def _resolve_single_output_path(
    args: argparse.Namespace,
    view_sequence: ViewSequence,
    *,
    result_source: str | list[str],
) -> Path:
    if args.output is not None:
        return Path(args.output)
    if args.output_root is not None:
        return _build_view_output_path(args.output_root, view_sequence, result_source=result_source)
    raise ValueError("Either --output or --output-root is required.")


def _build_view_output_path(
    output_root: str | Path,
    view_sequence: ViewSequence,
    *,
    result_source: str | list[str],
) -> Path:
    resolved_output_root = Path(output_root)
    relative_view_path = _resolve_output_relative_view_path(view_sequence, result_source=result_source)
    return resolved_output_root / relative_view_path / "view_temporal_fusion.json"


def _resolve_output_relative_view_path(view_sequence: ViewSequence, *, result_source: str | list[str]) -> Path:
    if isinstance(result_source, str):
        source_path = Path(result_source).resolve()
        frame_result_parent = view_sequence.frames[0].result_path.resolve().parent
        try:
            return frame_result_parent.relative_to(source_path)
        except ValueError:
            pass

    return _view_id_to_relative_path(view_sequence.view_id)


def _view_id_to_relative_path(view_id: str) -> Path:
    clean_parts = [part for part in view_id.split("/") if part not in {"", ".", ".."}]
    if not clean_parts:
        return Path("view")
    return Path(*clean_parts)


def _build_expected_output_paths(
    output_path: str | Path,
    *,
    write_video: bool,
    video_format: str,
) -> list[Path]:
    resolved_output_path = Path(output_path)
    expected_paths = [resolved_output_path, build_view_visualization_paths(resolved_output_path)["summary_png"]]
    if write_video:
        expected_paths.append(build_view_video_path(resolved_output_path, video_format=video_format))
    return expected_paths


def _is_view_fully_processed(
    output_path: str | Path,
    *,
    write_video: bool,
    video_format: str,
) -> bool:
    for path in _build_expected_output_paths(
        output_path,
        write_video=write_video,
        video_format=video_format,
    ):
        if not path.is_file():
            return False
        if path.stat().st_size <= 0:
            return False
    return True


def _run_single_view(
    view_sequence: ViewSequence,
    *,
    output_path: Path,
    min_supporting_frames: int,
    min_persistence_ratio: float,
    write_video: bool,
    video_fps: float,
    video_format: str,
    log=print,
) -> ViewLevelResult:
    log(f"Loaded view '{view_sequence.view_id}' with {view_sequence.frame_count} frame results.")

    view_result = run_temporal_fusion_on_view_sequence(
        view_sequence,
        min_supporting_frames=min_supporting_frames,
        min_persistence_ratio=min_persistence_ratio,
    )
    _print_fusion_summary(view_result, log=log)

    saved_output_path = save_view_level_result(view_result, output_path)
    visualization_paths = save_view_visualization_outputs(view_result, saved_output_path)
    log(f"Saved view-level result: {saved_output_path}")
    log(f"Saved summary visualization: {visualization_paths['summary_png']}")
    if write_video:
        video_path = save_view_demo_video(
            view_result,
            saved_output_path,
            fps=video_fps,
            video_format=video_format,
        )
        log(f"Saved demo video: {video_path}")

    return view_result


def _print_fusion_summary(view_result: ViewLevelResult, *, log=print) -> None:
    reference_selection = view_result.reference_selection
    fallback_count = sum(1 for registration in view_result.registrations if registration.fallback_used)
    success_count = len(view_result.registrations) - fallback_count

    log(
        f"Reference frame: {reference_selection.frame.image_name} "
        f"({reference_selection.method}, {reference_selection.score_name}={reference_selection.score_value:.0f})"
    )
    log(f"Registrations: {success_count} direct, {fallback_count} fallback.")
    log(f"Tracks: {len(view_result.tracks)} total, {view_result.persistent_lesion_count} persistent.")

    final_lesion = view_result.final_lesion
    if final_lesion is None:
        log("Final lesion: none.")
        return

    log(
        f"Final lesion: lesion {final_lesion.lesion_id} ({final_lesion.severity}), "
        f"median degree {final_lesion.median_degree:.3f}, "
        f"max degree {final_lesion.max_degree:.3f}, "
        f"persistence {final_lesion.persistence_ratio:.2%}."
    )


if __name__ == "__main__":
    raise SystemExit(main())
