from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

from .batch import BatchProcessSummary, discover_tree_jobs, process_tree
from .multiview import (
    LoadedMultiViewCase,
    LoadedMultiViewView,
    MultiViewFusionConfig,
    build_multiview_visualization_paths,
    load_multiview_case,
    run_multiview_fusion,
    save_multiview_case_result,
    save_multiview_visualization_outputs,
)
from .pipeline import PipelineConfig
from .temporal import (
    DEFAULT_MIN_PERSISTENCE_RATIO,
    DEFAULT_MIN_SUPPORTING_FRAMES,
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIEW_FRAME_COUNT,
    VIDEO_FORMATS,
    TemporalFusionConfig,
    ViewSequence,
    build_view_video_path,
    build_view_visualization_paths,
    load_view_sequences,
    run_temporal_fusion_variants_on_view_sequence,
    save_view_demo_video,
    save_view_level_result,
    save_view_visualization_outputs,
)
from .yolo.inference import (
    YoloBatchFailure,
    YoloInferenceConfig,
    discover_yolo_tree_jobs,
    process_yolo_tree,
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
class FrameSweepVariant:
    name: str
    config: PipelineConfig | YoloInferenceConfig
    detector: str = "vessel"


@dataclass(frozen=True, slots=True)
class TemporalSweepVariant:
    name: str
    config: TemporalFusionConfig


@dataclass(frozen=True, slots=True)
class TemporalSweepJob:
    frame_variant_name: str
    view_sequence: ViewSequence
    relative_view_path: Path
    output_root: Path
    temporal_variants: tuple[TemporalSweepVariant, ...]
    skip_existing: bool
    write_temporal_images: bool
    write_video: bool
    video_fps: float
    video_format: str


@dataclass(frozen=True, slots=True)
class TemporalSweepFailure:
    frame_variant_name: str
    view_id: str
    error: str


@dataclass(frozen=True, slots=True)
class TemporalSweepJobResult:
    frame_variant_name: str
    view_id: str
    processed_variants: int
    skipped_variants: int
    messages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TemporalSweepSummary:
    total_jobs: int
    workers: int
    processed_variants: int
    skipped_variants: int
    failed_jobs: int
    failures: tuple[TemporalSweepFailure, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_jobs": self.total_jobs,
            "workers": self.workers,
            "processed_variants": self.processed_variants,
            "skipped_variants": self.skipped_variants,
            "failed_jobs": self.failed_jobs,
            "failures": [asdict(failure) for failure in self.failures],
        }


@dataclass(frozen=True, slots=True)
class MultiViewSweepFailure:
    frame_variant_name: str
    temporal_variant_name: str
    case_input_path: str
    error: str


@dataclass(frozen=True, slots=True)
class MultiViewSweepSummary:
    total_jobs: int
    processed_cases: int
    skipped_cases: int
    failed_cases: int
    failures: tuple[MultiViewSweepFailure, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_jobs": self.total_jobs,
            "processed_cases": self.processed_cases,
            "skipped_cases": self.skipped_cases,
            "failed_cases": self.failed_cases,
            "failures": [asdict(failure) for failure in self.failures],
        }


@dataclass(frozen=True, slots=True)
class ParameterSweepResult:
    output_root: Path
    frame_results_root: Path
    temporal_results_root: Path
    multiview_results_root: Path | None
    frame_variants: tuple[FrameSweepVariant, ...]
    temporal_variants: tuple[TemporalSweepVariant, ...]
    frame_summary: BatchProcessSummary | "YoloFrameSweepSummary"
    temporal_summary: TemporalSweepSummary
    multiview_summary: MultiViewSweepSummary | None
    summary_json: Path


@dataclass(frozen=True, slots=True)
class YoloFrameSweepSummary:
    images_root: Path
    masks_root: Path | None
    output_root: Path
    total_jobs: int
    workers: int
    processed: int
    skipped_existing: int
    failed: int
    failures: tuple[YoloBatchFailure, ...]
    frame_variants: tuple[str, ...]
    variant_summaries: tuple[dict[str, Any], ...]
    review_images_saved: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": "yolo",
            "images_root": str(self.images_root),
            "masks_root": None if self.masks_root is None else str(self.masks_root),
            "output_root": str(self.output_root),
            "total_jobs": self.total_jobs,
            "workers": self.workers,
            "processed": self.processed,
            "skipped_existing": self.skipped_existing,
            "failed": self.failed,
            "failures": [asdict(failure) for failure in self.failures],
            "frame_variants": list(self.frame_variants),
            "variant_summaries": list(self.variant_summaries),
            "review_images_saved": self.review_images_saved,
        }


def run_parameter_sweep(
    *,
    images_root: str | Path,
    masks_root: str | Path | None = None,
    output_root: str | Path,
    frame_detector: str = "vessel",
    stenosis_thresholds: list[float] | None = None,
    average_radius_thresholds: list[float] | None = None,
    radius_outside_fraction_thresholds: list[float] | None = None,
    radius_min_outside_samples_values: list[int] | None = None,
    yolo_weights: str | Path | None = None,
    yolo_conf_thresholds: list[float] | None = None,
    yolo_iou_thresholds: list[float] | None = None,
    yolo_imgsz_values: list[int] | None = None,
    yolo_device: str | None = None,
    min_supporting_frames_values: list[int] | None = None,
    min_persistence_ratios: list[float] | None = None,
    base_config: PipelineConfig | None = None,
    workers: int = 1,
    temporal_workers: int = 1,
    allow_variable_frame_count: bool = False,
    expected_frame_count: int = DEFAULT_VIEW_FRAME_COUNT,
    skip_existing: bool = True,
    write_debug_images: bool = True,
    write_temporal_images: bool = True,
    write_video: bool = False,
    video_fps: float = DEFAULT_VIDEO_FPS,
    video_format: str = "mp4",
    run_multiview: bool = False,
    multiview_case_root_tree: str | Path | None = None,
    multiview_output_root: str | Path | None = None,
    multiview_config: MultiViewFusionConfig | None = None,
    split_multiview_by_coronary_side: bool = False,
    write_yolo_review_images: bool = False,
) -> ParameterSweepResult:
    if video_format not in VIDEO_FORMATS:
        raise ValueError(f"video_format must be one of {VIDEO_FORMATS}, got {video_format!r}.")
    if frame_detector not in {"vessel", "yolo"}:
        raise ValueError("frame_detector must be either 'vessel' or 'yolo'.")
    if frame_detector == "vessel" and masks_root is None:
        raise ValueError("masks_root is required when frame_detector='vessel'.")
    if frame_detector == "yolo" and yolo_weights is None:
        raise ValueError("yolo_weights is required when frame_detector='yolo'.")

    resolved_output_root = Path(output_root)
    frame_results_root = resolved_output_root / "frame_results"
    temporal_results_root = resolved_output_root / "temporal_results"
    pipeline_config = base_config or PipelineConfig()
    if frame_detector == "vessel":
        frame_variants = build_frame_sweep_variants(
            pipeline_config,
            stenosis_thresholds=stenosis_thresholds,
            average_radius_thresholds=average_radius_thresholds,
            radius_outside_fraction_thresholds=radius_outside_fraction_thresholds,
            radius_min_outside_samples_values=radius_min_outside_samples_values,
        )
    else:
        frame_variants = build_yolo_frame_sweep_variants(
            yolo_weights=yolo_weights,
            yolo_conf_thresholds=yolo_conf_thresholds,
            yolo_iou_thresholds=yolo_iou_thresholds,
            yolo_imgsz_values=yolo_imgsz_values,
            device=yolo_device,
            mask_threshold=pipeline_config.mask_threshold,
        )
    temporal_variants = build_temporal_sweep_variants(
        min_supporting_frames_values=min_supporting_frames_values,
        min_persistence_ratios=min_persistence_ratios,
    )

    if frame_detector == "vessel":
        assert masks_root is not None
        jobs = discover_tree_jobs(images_root, masks_root)
        frame_summary = process_tree(
            jobs,
            frame_results_root,
            images_root=images_root,
            masks_root=masks_root,
            config=frame_variants[0].config,
            skip_existing=skip_existing,
            workers=workers,
            threshold_variants=[(variant.name, variant.config) for variant in frame_variants],
            write_debug_images=write_debug_images,
        )
    else:
        frame_summary = run_yolo_frame_sweep(
            images_root=images_root,
            masks_root=masks_root,
            frame_results_root=frame_results_root,
            frame_variants=frame_variants,
            skip_existing=skip_existing,
            workers=workers,
            write_review_images=write_yolo_review_images,
        )
    temporal_summary = run_temporal_sweep(
        frame_results_root=frame_results_root,
        temporal_results_root=temporal_results_root,
        frame_variants=frame_variants,
        temporal_variants=temporal_variants,
        expected_frame_count=None if allow_variable_frame_count else expected_frame_count,
        skip_existing=skip_existing,
        workers=temporal_workers,
        write_temporal_images=write_temporal_images,
        write_video=write_video,
        video_fps=video_fps,
        video_format=video_format,
    )
    resolved_multiview_results_root = None
    multiview_summary = None
    if run_multiview:
        resolved_multiview_results_root = (
            Path(multiview_output_root)
            if multiview_output_root is not None
            else resolved_output_root / "multiview_results"
        )
        multiview_summary = run_multiview_sweep(
            case_root_tree=Path(images_root) if multiview_case_root_tree is None else Path(multiview_case_root_tree),
            temporal_results_root=temporal_results_root,
            output_root=resolved_multiview_results_root,
            frame_variants=frame_variants,
            temporal_variants=temporal_variants,
            config=MultiViewFusionConfig() if multiview_config is None else multiview_config,
            split_by_coronary_side=split_multiview_by_coronary_side,
            skip_existing=skip_existing,
        )

    summary_json = resolved_output_root / "parameter_sweep_summary.json"
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(
            _sweep_summary_payload(
                images_root=Path(images_root),
                masks_root=None if masks_root is None else Path(masks_root),
                output_root=resolved_output_root,
                frame_results_root=frame_results_root,
                temporal_results_root=temporal_results_root,
                multiview_results_root=resolved_multiview_results_root,
                frame_variants=frame_variants,
                temporal_variants=temporal_variants,
                frame_summary=frame_summary,
                temporal_summary=temporal_summary,
                multiview_summary=multiview_summary,
                allow_variable_frame_count=allow_variable_frame_count,
                expected_frame_count=expected_frame_count,
                skip_existing=skip_existing,
                write_debug_images=write_debug_images,
                write_temporal_images=write_temporal_images,
                write_video=write_video,
                video_fps=video_fps,
                video_format=video_format,
                run_multiview=run_multiview,
                multiview_case_root_tree=(
                    Path(images_root) if multiview_case_root_tree is None else Path(multiview_case_root_tree)
                ),
                multiview_config=MultiViewFusionConfig() if multiview_config is None else multiview_config,
                split_multiview_by_coronary_side=split_multiview_by_coronary_side,
                frame_detector=frame_detector,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return ParameterSweepResult(
        output_root=resolved_output_root,
        frame_results_root=frame_results_root,
        temporal_results_root=temporal_results_root,
        multiview_results_root=resolved_multiview_results_root,
        frame_variants=tuple(frame_variants),
        temporal_variants=tuple(temporal_variants),
        frame_summary=frame_summary,
        temporal_summary=temporal_summary,
        multiview_summary=multiview_summary,
        summary_json=summary_json,
    )


def build_frame_sweep_variants(
    base_config: PipelineConfig,
    *,
    stenosis_thresholds: list[float] | None = None,
    average_radius_thresholds: list[float] | None = None,
    radius_outside_fraction_thresholds: list[float] | None = None,
    radius_min_outside_samples_values: list[int] | None = None,
) -> list[FrameSweepVariant]:
    resolved_stenosis_thresholds = stenosis_thresholds or [base_config.stenosis_threshold]
    resolved_average_radius_thresholds = average_radius_thresholds or [base_config.average_radius_threshold]
    resolved_radius_outside_fraction_thresholds = (
        radius_outside_fraction_thresholds or [base_config.radius_outside_fraction_threshold]
    )
    resolved_radius_min_outside_samples_values = (
        radius_min_outside_samples_values or [base_config.radius_min_outside_samples]
    )

    variants: list[FrameSweepVariant] = []
    for radius_outside_fraction_threshold in resolved_radius_outside_fraction_thresholds:
        if radius_outside_fraction_threshold < 0.0:
            raise ValueError("radius_outside_fraction_threshold values must be >= 0.0.")
        for radius_min_outside_samples in resolved_radius_min_outside_samples_values:
            if radius_min_outside_samples < 1:
                raise ValueError("radius_min_outside_samples values must be >= 1.")
            for stenosis_threshold in resolved_stenosis_thresholds:
                for average_radius_threshold in resolved_average_radius_thresholds:
                    config = PipelineConfig(
                        **{
                            **asdict(base_config),
                            "radius_outside_fraction_threshold": radius_outside_fraction_threshold,
                            "radius_min_outside_samples": radius_min_outside_samples,
                            "stenosis_threshold": stenosis_threshold,
                            "average_radius_threshold": average_radius_threshold,
                        }
                    )
                    variants.append(
                        FrameSweepVariant(
                            name=frame_variant_name(config),
                            config=config,
                        )
                    )
    _validate_unique_names([variant.name for variant in variants], "frame variant")
    return variants


def build_yolo_frame_sweep_variants(
    *,
    yolo_weights: str | Path | None,
    yolo_conf_thresholds: list[float] | None = None,
    yolo_iou_thresholds: list[float] | None = None,
    yolo_imgsz_values: list[int] | None = None,
    device: str | None = None,
    mask_threshold: int | None = None,
) -> list[FrameSweepVariant]:
    if yolo_weights is None:
        raise ValueError("yolo_weights is required for YOLO frame sweeps.")

    resolved_conf_thresholds = yolo_conf_thresholds or [0.25]
    resolved_iou_thresholds = yolo_iou_thresholds or [0.70]
    resolved_imgsz_values = yolo_imgsz_values or [1024]

    variants: list[FrameSweepVariant] = []
    for conf_threshold in resolved_conf_thresholds:
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError("yolo_conf_threshold values must be in the range [0.0, 1.0].")
        for iou_threshold in resolved_iou_thresholds:
            if not 0.0 <= iou_threshold <= 1.0:
                raise ValueError("yolo_iou_threshold values must be in the range [0.0, 1.0].")
            for imgsz in resolved_imgsz_values:
                if imgsz < 1:
                    raise ValueError("yolo_imgsz values must be >= 1.")
                config = YoloInferenceConfig(
                    weights=str(yolo_weights),
                    imgsz=int(imgsz),
                    conf=float(conf_threshold),
                    iou=float(iou_threshold),
                    device=device,
                    mask_threshold=PipelineConfig().mask_threshold if mask_threshold is None else mask_threshold,
                )
                variants.append(
                    FrameSweepVariant(
                        name=yolo_frame_variant_name(config),
                        config=config,
                        detector="yolo",
                    )
                )
    _validate_unique_names([variant.name for variant in variants], "YOLO frame variant")
    return variants


def build_temporal_sweep_variants(
    *,
    min_supporting_frames_values: list[int] | None = None,
    min_persistence_ratios: list[float] | None = None,
) -> list[TemporalSweepVariant]:
    resolved_min_supporting_frames_values = min_supporting_frames_values or [DEFAULT_MIN_SUPPORTING_FRAMES]
    resolved_min_persistence_ratios = min_persistence_ratios or [DEFAULT_MIN_PERSISTENCE_RATIO]

    variants: list[TemporalSweepVariant] = []
    for min_supporting_frames in resolved_min_supporting_frames_values:
        if min_supporting_frames < 1:
            raise ValueError("min_supporting_frames values must be >= 1.")
        for min_persistence_ratio in resolved_min_persistence_ratios:
            if not 0.0 < min_persistence_ratio <= 1.0:
                raise ValueError("min_persistence_ratio values must be in the range (0.0, 1.0].")
            config = TemporalFusionConfig(
                min_supporting_frames=min_supporting_frames,
                min_persistence_ratio=min_persistence_ratio,
            )
            variants.append(
                TemporalSweepVariant(
                    name=temporal_variant_name(config),
                    config=config,
                )
            )
    _validate_unique_names([variant.name for variant in variants], "temporal variant")
    return variants


def run_yolo_frame_sweep(
    *,
    images_root: str | Path,
    masks_root: str | Path | None,
    frame_results_root: str | Path,
    frame_variants: list[FrameSweepVariant],
    skip_existing: bool,
    workers: int,
    write_review_images: bool,
) -> YoloFrameSweepSummary:
    yolo_variants = [variant for variant in frame_variants if variant.detector == "yolo"]
    if len(yolo_variants) != len(frame_variants):
        raise ValueError("run_yolo_frame_sweep received non-YOLO frame variants.")

    resolved_frame_results_root = Path(frame_results_root)
    jobs = discover_yolo_tree_jobs(images_root, masks_root=masks_root)
    variant_summaries: list[dict[str, Any]] = []
    failures: list[YoloBatchFailure] = []
    total_jobs = 0
    processed = 0
    skipped_existing = 0
    failed = 0
    resolved_workers = 1

    for variant in yolo_variants:
        if not isinstance(variant.config, YoloInferenceConfig):
            raise ValueError(f"YOLO frame variant {variant.name!r} has an invalid config type.")
        summary = process_yolo_tree(
            jobs,
            output_root=resolved_frame_results_root / variant.name,
            images_root=images_root,
            masks_root=masks_root,
            config=variant.config,
            skip_existing=skip_existing,
            workers=workers,
            write_review_images=write_review_images,
        )
        summary_payload = summary.to_dict()
        summary_payload["variant_name"] = variant.name
        variant_summaries.append(summary_payload)
        total_jobs += summary.total_jobs
        processed += summary.processed
        skipped_existing += summary.skipped_existing
        failed += summary.failed
        failures.extend(summary.failures)
        resolved_workers = summary.workers

    return YoloFrameSweepSummary(
        images_root=Path(images_root),
        masks_root=None if masks_root is None else Path(masks_root),
        output_root=resolved_frame_results_root,
        total_jobs=total_jobs,
        workers=resolved_workers,
        processed=processed,
        skipped_existing=skipped_existing,
        failed=failed,
        failures=tuple(failures),
        frame_variants=tuple(variant.name for variant in yolo_variants),
        variant_summaries=tuple(variant_summaries),
        review_images_saved=write_review_images,
    )


def run_temporal_sweep(
    *,
    frame_results_root: str | Path,
    temporal_results_root: str | Path,
    frame_variants: list[FrameSweepVariant],
    temporal_variants: list[TemporalSweepVariant],
    expected_frame_count: int | None,
    skip_existing: bool,
    workers: int,
    write_video: bool,
    write_temporal_images: bool,
    video_fps: float,
    video_format: str,
) -> TemporalSweepSummary:
    worker_count = _resolve_worker_count(workers)
    resolved_frame_results_root = Path(frame_results_root)
    resolved_temporal_results_root = Path(temporal_results_root)
    jobs: list[TemporalSweepJob] = []

    for frame_variant in frame_variants:
        frame_variant_root = resolved_frame_results_root / frame_variant.name
        view_sequences = load_view_sequences(
            frame_variant_root,
            expected_frame_count=expected_frame_count,
        )
        for view_sequence in view_sequences:
            jobs.append(
                TemporalSweepJob(
                    frame_variant_name=frame_variant.name,
                    view_sequence=view_sequence,
                    relative_view_path=_resolve_relative_view_path(view_sequence, frame_variant_root),
                    output_root=resolved_temporal_results_root / frame_variant.name,
                    temporal_variants=tuple(temporal_variants),
                    skip_existing=skip_existing,
                    write_temporal_images=write_temporal_images,
                    write_video=write_video,
                    video_fps=video_fps,
                    video_format=video_format,
                )
            )

    processed_variants = 0
    skipped_variants = 0
    failures: list[TemporalSweepFailure] = []

    with tqdm(total=len(jobs), desc="Sweeping temporal fusion", unit="view", dynamic_ncols=True) as progress:
        if worker_count == 1 or len(jobs) <= 1:
            for job in jobs:
                try:
                    result = _run_temporal_sweep_job(job)
                except Exception as exc:  # pragma: no cover - protects long sweeps.
                    failures.append(_temporal_failure(job, exc))
                    progress.write(f"Failed temporal sweep: {job.frame_variant_name}/{job.view_sequence.view_id} -> {exc}")
                    progress.update(1)
                    _set_temporal_progress(progress, processed_variants, skipped_variants, len(failures))
                    continue

                _write_job_messages(result, log=progress.write)
                processed_variants += result.processed_variants
                skipped_variants += result.skipped_variants
                progress.update(1)
                _set_temporal_progress(progress, processed_variants, skipped_variants, len(failures))
        elif jobs:
            with ProcessPoolExecutor(max_workers=min(worker_count, len(jobs))) as executor:
                future_to_job = {executor.submit(_run_temporal_sweep_job, job): job for job in jobs}
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover - protects parent process progress.
                        failures.append(_temporal_failure(job, exc))
                        progress.write(f"Failed temporal sweep: {job.frame_variant_name}/{job.view_sequence.view_id} -> {exc}")
                        progress.update(1)
                        _set_temporal_progress(progress, processed_variants, skipped_variants, len(failures))
                        continue

                    _write_job_messages(result, log=progress.write)
                    processed_variants += result.processed_variants
                    skipped_variants += result.skipped_variants
                    progress.update(1)
                    _set_temporal_progress(progress, processed_variants, skipped_variants, len(failures))

    return TemporalSweepSummary(
        total_jobs=len(jobs),
        workers=worker_count,
        processed_variants=processed_variants,
        skipped_variants=skipped_variants,
        failed_jobs=len(failures),
        failures=tuple(failures),
    )


def run_multiview_sweep(
    *,
    case_root_tree: str | Path,
    temporal_results_root: str | Path,
    output_root: str | Path,
    frame_variants: list[FrameSweepVariant],
    temporal_variants: list[TemporalSweepVariant],
    config: MultiViewFusionConfig,
    split_by_coronary_side: bool,
    skip_existing: bool,
) -> MultiViewSweepSummary:
    resolved_case_root_tree = Path(case_root_tree)
    resolved_temporal_results_root = Path(temporal_results_root)
    resolved_output_root = Path(output_root)
    case_input_paths = _discover_multiview_case_input_paths(resolved_case_root_tree)
    jobs = [
        (frame_variant, temporal_variant, case_input_path)
        for frame_variant in frame_variants
        for temporal_variant in temporal_variants
        for case_input_path in case_input_paths
    ]

    processed_cases = 0
    skipped_cases = 0
    failures: list[MultiViewSweepFailure] = []

    with tqdm(total=len(jobs), desc="Sweeping multi-view fusion", unit="case", dynamic_ncols=True) as progress:
        for frame_variant, temporal_variant, case_input_path in jobs:
            temporal_variant_root = resolved_temporal_results_root / frame_variant.name / temporal_variant.name
            output_path = _multiview_output_path(
                case_input_path,
                case_root_tree=resolved_case_root_tree,
                output_root=resolved_output_root / frame_variant.name / temporal_variant.name,
            )
            if skip_existing and _is_multiview_output_complete(
                output_path,
                split_by_coronary_side=split_by_coronary_side,
            ):
                skipped_cases += 1
                progress.update(1)
                _set_multiview_progress(progress, processed_cases, skipped_cases, len(failures))
                continue

            try:
                multiview_case = load_multiview_case(
                    case_input_path,
                    temporal_results_root=temporal_variant_root,
                    case_root_tree=resolved_case_root_tree,
                )
                if split_by_coronary_side:
                    _save_split_multiview_case_result(multiview_case, output_path, config)
                else:
                    case_result = run_multiview_fusion(multiview_case, config=config)
                    saved_path = save_multiview_case_result(case_result, output_path)
                    save_multiview_visualization_outputs(case_result, saved_path)
            except Exception as exc:  # pragma: no cover - protects long sweeps.
                failures.append(
                    MultiViewSweepFailure(
                        frame_variant_name=frame_variant.name,
                        temporal_variant_name=temporal_variant.name,
                        case_input_path=str(case_input_path),
                        error=str(exc),
                    )
                )
                progress.write(
                    "Failed multi-view sweep: "
                    f"{frame_variant.name}/{temporal_variant.name}/{case_input_path} -> {exc}"
                )
            else:
                processed_cases += 1

            progress.update(1)
            _set_multiview_progress(progress, processed_cases, skipped_cases, len(failures))

    return MultiViewSweepSummary(
        total_jobs=len(jobs),
        processed_cases=processed_cases,
        skipped_cases=skipped_cases,
        failed_cases=len(failures),
        failures=tuple(failures),
    )


def frame_variant_name(config: PipelineConfig) -> str:
    return (
        f"radius_outside_fraction_threshold_{_format_value(config.radius_outside_fraction_threshold)}"
        f"__radius_min_outside_samples_{config.radius_min_outside_samples}"
        f"__stenosis_threshold_{_format_value(config.stenosis_threshold)}"
        f"__average_radius_threshold_{_format_value(config.average_radius_threshold)}"
    )


def yolo_frame_variant_name(config: YoloInferenceConfig) -> str:
    return (
        f"detector_yolo"
        f"__yolo_conf_{_format_yolo_threshold(config.conf)}"
        f"__yolo_iou_{_format_yolo_threshold(config.iou)}"
        f"__yolo_imgsz_{config.imgsz}"
    )


def temporal_variant_name(config: TemporalFusionConfig) -> str:
    return (
        f"min_supporting_frames_{config.min_supporting_frames}"
        f"__min_persistence_ratio_{_format_value(config.min_persistence_ratio)}"
    )


def _run_temporal_sweep_job(job: TemporalSweepJob) -> TemporalSweepJobResult:
    pending_variants: list[TemporalSweepVariant] = []
    skipped_variants = 0
    for temporal_variant in job.temporal_variants:
        output_path = _temporal_output_path(job, temporal_variant)
        if job.skip_existing and _is_temporal_variant_complete(
            output_path,
            write_temporal_images=job.write_temporal_images,
            write_video=job.write_video,
            video_format=job.video_format,
        ):
            skipped_variants += 1
            continue
        pending_variants.append(temporal_variant)

    if not pending_variants:
        return TemporalSweepJobResult(
            frame_variant_name=job.frame_variant_name,
            view_id=job.view_sequence.view_id,
            processed_variants=0,
            skipped_variants=skipped_variants,
            messages=(),
        )

    results = run_temporal_fusion_variants_on_view_sequence(
        job.view_sequence,
        [variant.config for variant in pending_variants],
    )
    messages: list[str] = []
    for temporal_variant, view_result in zip(pending_variants, results, strict=True):
        output_path = _temporal_output_path(job, temporal_variant)
        saved_output_path = save_view_level_result(view_result, output_path)
        messages.append(f"Saved temporal sweep result: {saved_output_path}")
        if job.write_temporal_images:
            visualization_paths = save_view_visualization_outputs(view_result, saved_output_path)
            messages.append(f"Saved temporal sweep summary: {visualization_paths['summary_png']}")
        if job.write_video:
            video_path = save_view_demo_video(
                view_result,
                saved_output_path,
                fps=job.video_fps,
                video_format=job.video_format,
            )
            messages.append(f"Saved temporal sweep video: {video_path}")

    return TemporalSweepJobResult(
        frame_variant_name=job.frame_variant_name,
        view_id=job.view_sequence.view_id,
        processed_variants=len(pending_variants),
        skipped_variants=skipped_variants,
        messages=tuple(messages),
    )


def _temporal_output_path(job: TemporalSweepJob, temporal_variant: TemporalSweepVariant) -> Path:
    return job.output_root / temporal_variant.name / job.relative_view_path / "view_temporal_fusion.json"


def _is_temporal_variant_complete(
    output_path: Path,
    *,
    write_temporal_images: bool,
    write_video: bool,
    video_format: str,
) -> bool:
    expected_paths = [output_path]
    if write_temporal_images:
        expected_paths.append(build_view_visualization_paths(output_path)["summary_png"])
    if write_video:
        expected_paths.append(build_view_video_path(output_path, video_format=video_format))
    return all(path.is_file() and path.stat().st_size > 0 for path in expected_paths)


def _discover_multiview_case_input_paths(case_root_tree: Path) -> list[Path]:
    if not case_root_tree.exists():
        raise FileNotFoundError(f"Multi-view case root tree does not exist: {case_root_tree}")
    if not case_root_tree.is_dir():
        raise NotADirectoryError(f"Multi-view case root tree is not a directory: {case_root_tree}")

    case_input_paths = sorted(path for path in case_root_tree.rglob("views.json") if path.is_file())
    if not case_input_paths:
        raise FileNotFoundError(f"No views.json files were found under: {case_root_tree}")
    return case_input_paths


def _multiview_output_path(case_input_path: Path, *, case_root_tree: Path, output_root: Path) -> Path:
    relative_case_dir = case_input_path.parent.resolve().relative_to(case_root_tree.resolve())
    return output_root / relative_case_dir / "case_multiview_fusion.json"


def _is_multiview_output_complete(output_path: Path, *, split_by_coronary_side: bool) -> bool:
    expected_paths = [output_path]
    if not split_by_coronary_side:
        visualization_paths = build_multiview_visualization_paths(output_path)
        expected_paths.extend(
            [
                visualization_paths["summary_png"],
                visualization_paths["support_matrix_png"],
            ]
        )
    return all(path.is_file() and path.stat().st_size > 0 for path in expected_paths)


def _save_split_multiview_case_result(
    multiview_case: LoadedMultiViewCase,
    output_path: Path,
    config: MultiViewFusionConfig,
) -> Path:
    side_groups = _split_views_by_coronary_side(multiview_case)
    side_results: dict[str, object] = {}
    skipped_sides: list[str] = []
    for side in ("left", "right"):
        side_views = side_groups[side]
        if not side_views:
            skipped_sides.append(side)
            continue
        side_case = LoadedMultiViewCase(
            case_id=f"{multiview_case.case_id}:{side}",
            views=side_views,
        )
        side_results[side] = run_multiview_fusion(side_case, config=config).to_dict()

    payload: dict[str, object] = {
        "case_id": multiview_case.case_id,
        "split_by_coronary_side": True,
        "view_diversity_mode": config.view_diversity_mode,
        "side_results": side_results,
        "unknown_views": [view.view_input.to_dict() for view in side_groups["unknown"]],
        "skipped_sides": skipped_sides,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _split_views_by_coronary_side(multiview_case: LoadedMultiViewCase) -> dict[str, list[LoadedMultiViewView]]:
    groups: dict[str, list[LoadedMultiViewView]] = {"left": [], "right": [], "unknown": []}
    for view in multiview_case.views:
        side = (view.view_input.coronary_side or "unknown").strip().lower()
        if side not in {"left", "right"}:
            side = "unknown"
        groups[side].append(view)
    return groups


def _resolve_relative_view_path(view_sequence: ViewSequence, frame_variant_root: Path) -> Path:
    frame_result_parent = view_sequence.frames[0].result_path.resolve().parent
    try:
        return frame_result_parent.relative_to(frame_variant_root.resolve())
    except ValueError:
        clean_parts = [part for part in view_sequence.view_id.split("/") if part not in {"", ".", ".."}]
        return Path(*clean_parts) if clean_parts else Path("view")


def _resolve_worker_count(workers: int) -> int:
    if workers < 0:
        raise ValueError("workers must be 0 or greater.")
    if workers == 0:
        return max(1, os.cpu_count() or 1)
    return workers


def _temporal_failure(job: TemporalSweepJob, exc: Exception) -> TemporalSweepFailure:
    return TemporalSweepFailure(
        frame_variant_name=job.frame_variant_name,
        view_id=job.view_sequence.view_id,
        error=str(exc),
    )


def _set_temporal_progress(progress, processed_variants: int, skipped_variants: int, failed_jobs: int) -> None:
    progress.set_postfix(
        processed_variants=processed_variants,
        skipped_variants=skipped_variants,
        failed_jobs=failed_jobs,
    )


def _set_multiview_progress(progress, processed_cases: int, skipped_cases: int, failed_cases: int) -> None:
    progress.set_postfix(
        processed_cases=processed_cases,
        skipped_cases=skipped_cases,
        failed_cases=failed_cases,
    )


def _write_job_messages(result: TemporalSweepJobResult, *, log=print) -> None:
    for message in result.messages:
        log(message)


def _sweep_summary_payload(
    *,
    images_root: Path,
    masks_root: Path | None,
    output_root: Path,
    frame_results_root: Path,
    temporal_results_root: Path,
    multiview_results_root: Path | None,
    frame_variants: list[FrameSweepVariant],
    temporal_variants: list[TemporalSweepVariant],
    frame_summary: BatchProcessSummary,
    temporal_summary: TemporalSweepSummary,
    multiview_summary: MultiViewSweepSummary | None,
    allow_variable_frame_count: bool,
    expected_frame_count: int,
    skip_existing: bool,
    write_debug_images: bool,
    write_temporal_images: bool,
    write_video: bool,
    video_fps: float,
    video_format: str,
    run_multiview: bool,
    multiview_case_root_tree: Path,
    multiview_config: MultiViewFusionConfig,
    split_multiview_by_coronary_side: bool,
    frame_detector: str,
) -> dict[str, Any]:
    return {
        "inputs": {
            "images_root": str(images_root),
            "masks_root": None if masks_root is None else str(masks_root),
        },
        "outputs": {
            "output_root": str(output_root),
            "frame_results_root": str(frame_results_root),
            "temporal_results_root": str(temporal_results_root),
            "multiview_results_root": None if multiview_results_root is None else str(multiview_results_root),
        },
        "config": {
            "frame_detector": frame_detector,
            "allow_variable_frame_count": allow_variable_frame_count,
            "expected_frame_count": expected_frame_count,
            "skip_existing": skip_existing,
            "write_debug_images": write_debug_images,
            "write_temporal_images": write_temporal_images,
            "write_video": write_video,
            "video_fps": video_fps,
            "video_format": video_format,
            "run_multiview": run_multiview,
            "multiview_case_root_tree": str(multiview_case_root_tree),
            "multiview_config": multiview_config.to_dict(),
            "split_multiview_by_coronary_side": split_multiview_by_coronary_side,
        },
        "frame_variants": [
            {"name": variant.name, "detector": variant.detector, "config": asdict(variant.config)}
            for variant in frame_variants
        ],
        "temporal_variants": [
            {"name": variant.name, "config": asdict(variant.config)}
            for variant in temporal_variants
        ],
        "frame_summary": frame_summary.to_dict(),
        "temporal_summary": temporal_summary.to_dict(),
        "multiview_summary": None if multiview_summary is None else multiview_summary.to_dict(),
    }


def _format_value(value: float | int) -> str:
    return f"{value:g}".replace("-", "minus_").replace(".", "p")


def _format_yolo_threshold(value: float) -> str:
    return f"{value:.2f}".replace("-", "minus_").replace(".", "p")


def _validate_unique_names(names: list[str], label: str) -> None:
    if len(names) == len(set(names)):
        return
    duplicates = sorted({name for name in names if names.count(name) > 1})
    raise ValueError(f"Duplicate {label} names after formatting: {duplicates}")
