from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

from .batch import BatchProcessSummary, discover_tree_jobs, process_tree
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
    config: PipelineConfig


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
class ParameterSweepResult:
    output_root: Path
    frame_results_root: Path
    temporal_results_root: Path
    frame_variants: tuple[FrameSweepVariant, ...]
    temporal_variants: tuple[TemporalSweepVariant, ...]
    frame_summary: BatchProcessSummary
    temporal_summary: TemporalSweepSummary
    summary_json: Path


def run_parameter_sweep(
    *,
    images_root: str | Path,
    masks_root: str | Path,
    output_root: str | Path,
    stenosis_thresholds: list[float] | None = None,
    average_radius_thresholds: list[float] | None = None,
    radius_outside_fraction_thresholds: list[float] | None = None,
    radius_min_outside_samples_values: list[int] | None = None,
    min_supporting_frames_values: list[int] | None = None,
    min_persistence_ratios: list[float] | None = None,
    base_config: PipelineConfig | None = None,
    workers: int = 1,
    temporal_workers: int = 1,
    allow_variable_frame_count: bool = False,
    expected_frame_count: int = DEFAULT_VIEW_FRAME_COUNT,
    skip_existing: bool = True,
    write_debug_images: bool = True,
    write_video: bool = False,
    video_fps: float = DEFAULT_VIDEO_FPS,
    video_format: str = "mp4",
) -> ParameterSweepResult:
    if video_format not in VIDEO_FORMATS:
        raise ValueError(f"video_format must be one of {VIDEO_FORMATS}, got {video_format!r}.")

    resolved_output_root = Path(output_root)
    frame_results_root = resolved_output_root / "frame_results"
    temporal_results_root = resolved_output_root / "temporal_results"
    pipeline_config = base_config or PipelineConfig()
    frame_variants = build_frame_sweep_variants(
        pipeline_config,
        stenosis_thresholds=stenosis_thresholds,
        average_radius_thresholds=average_radius_thresholds,
        radius_outside_fraction_thresholds=radius_outside_fraction_thresholds,
        radius_min_outside_samples_values=radius_min_outside_samples_values,
    )
    temporal_variants = build_temporal_sweep_variants(
        min_supporting_frames_values=min_supporting_frames_values,
        min_persistence_ratios=min_persistence_ratios,
    )

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
    temporal_summary = run_temporal_sweep(
        frame_results_root=frame_results_root,
        temporal_results_root=temporal_results_root,
        frame_variants=frame_variants,
        temporal_variants=temporal_variants,
        expected_frame_count=None if allow_variable_frame_count else expected_frame_count,
        skip_existing=skip_existing,
        workers=temporal_workers,
        write_video=write_video,
        video_fps=video_fps,
        video_format=video_format,
    )

    summary_json = resolved_output_root / "parameter_sweep_summary.json"
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(
            _sweep_summary_payload(
                images_root=Path(images_root),
                masks_root=Path(masks_root),
                output_root=resolved_output_root,
                frame_results_root=frame_results_root,
                temporal_results_root=temporal_results_root,
                frame_variants=frame_variants,
                temporal_variants=temporal_variants,
                frame_summary=frame_summary,
                temporal_summary=temporal_summary,
                allow_variable_frame_count=allow_variable_frame_count,
                expected_frame_count=expected_frame_count,
                skip_existing=skip_existing,
                write_debug_images=write_debug_images,
                write_video=write_video,
                video_fps=video_fps,
                video_format=video_format,
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
        frame_variants=tuple(frame_variants),
        temporal_variants=tuple(temporal_variants),
        frame_summary=frame_summary,
        temporal_summary=temporal_summary,
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


def frame_variant_name(config: PipelineConfig) -> str:
    return (
        f"radius_outside_fraction_threshold_{_format_value(config.radius_outside_fraction_threshold)}"
        f"__radius_min_outside_samples_{config.radius_min_outside_samples}"
        f"__stenosis_threshold_{_format_value(config.stenosis_threshold)}"
        f"__average_radius_threshold_{_format_value(config.average_radius_threshold)}"
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
        visualization_paths = save_view_visualization_outputs(view_result, saved_output_path)
        messages.append(f"Saved temporal sweep result: {saved_output_path}")
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


def _is_temporal_variant_complete(output_path: Path, *, write_video: bool, video_format: str) -> bool:
    expected_paths = [output_path, build_view_visualization_paths(output_path)["summary_png"]]
    if write_video:
        expected_paths.append(build_view_video_path(output_path, video_format=video_format))
    return all(path.is_file() and path.stat().st_size > 0 for path in expected_paths)


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


def _write_job_messages(result: TemporalSweepJobResult, *, log=print) -> None:
    for message in result.messages:
        log(message)


def _sweep_summary_payload(
    *,
    images_root: Path,
    masks_root: Path,
    output_root: Path,
    frame_results_root: Path,
    temporal_results_root: Path,
    frame_variants: list[FrameSweepVariant],
    temporal_variants: list[TemporalSweepVariant],
    frame_summary: BatchProcessSummary,
    temporal_summary: TemporalSweepSummary,
    allow_variable_frame_count: bool,
    expected_frame_count: int,
    skip_existing: bool,
    write_debug_images: bool,
    write_video: bool,
    video_fps: float,
    video_format: str,
) -> dict[str, Any]:
    return {
        "inputs": {
            "images_root": str(images_root),
            "masks_root": str(masks_root),
        },
        "outputs": {
            "output_root": str(output_root),
            "frame_results_root": str(frame_results_root),
            "temporal_results_root": str(temporal_results_root),
        },
        "config": {
            "allow_variable_frame_count": allow_variable_frame_count,
            "expected_frame_count": expected_frame_count,
            "skip_existing": skip_existing,
            "write_debug_images": write_debug_images,
            "write_video": write_video,
            "video_fps": video_fps,
            "video_format": video_format,
        },
        "frame_variants": [
            {"name": variant.name, "config": asdict(variant.config)}
            for variant in frame_variants
        ],
        "temporal_variants": [
            {"name": variant.name, "config": asdict(variant.config)}
            for variant in temporal_variants
        ],
        "frame_summary": frame_summary.to_dict(),
        "temporal_summary": temporal_summary.to_dict(),
    }


def _format_value(value: float | int) -> str:
    return f"{value:g}".replace("-", "minus_").replace(".", "p")


def _validate_unique_names(names: list[str], label: str) -> None:
    if len(names) == len(set(names)):
        return
    duplicates = sorted({name for name in names if names.count(name) > 1})
    raise ValueError(f"Duplicate {label} names after formatting: {duplicates}")
