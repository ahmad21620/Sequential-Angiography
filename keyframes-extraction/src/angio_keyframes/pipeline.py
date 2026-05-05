from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from angio_keyframes.backends import (
    BACKEND_CHOICES,
    BackendName,
    ScoringBackend,
    build_baseline_image,
    compute_contrast_fill_score,
    create_backend,
)
from angio_keyframes.cadica import read_cadica_selected_frame_count, resolve_cadica_selected_videos_root
from angio_keyframes.discovery import discover_frame_directories
from angio_keyframes.images import list_image_files, load_grayscale_image, write_grayscale_image
from angio_keyframes.models import ExtractionResult, KeyframeCandidate

ExtractionJob = tuple[BackendName, Path, Path, int, int, int, bool, bool]
PATIENT_METADATA_FILENAMES = ("views.json", "patient.json")


def smooth_score_curve(scores: np.ndarray, smoothing_window: int) -> np.ndarray:
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be greater than 0.")
    if smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be odd for centered smoothing.")
    if scores.size == 0:
        return np.array([], dtype=np.float32)

    kernel = np.ones(smoothing_window, dtype=np.float32)
    radius = smoothing_window // 2
    score_sums = np.convolve(scores.astype(np.float32, copy=False), kernel, mode="full")
    score_counts = np.convolve(np.ones(scores.shape[0], dtype=np.float32), kernel, mode="full")
    smoothed_scores = score_sums[radius : radius + scores.shape[0]]
    smoothed_scores = smoothed_scores / score_counts[radius : radius + scores.shape[0]]

    return smoothed_scores.astype(np.float32, copy=False)


def select_keyframe_window(
    candidates: list[KeyframeCandidate],
    limit: int,
    smoothing_window: int,
) -> list[KeyframeCandidate]:
    if limit <= 0:
        raise ValueError("limit must be greater than 0.")
    if not candidates:
        return []

    window_size = min(limit, len(candidates))
    scores = np.array([candidate.score for candidate in candidates], dtype=np.float32)
    smoothed_scores = smooth_score_curve(scores, smoothing_window)
    peak_index = int(np.argmax(smoothed_scores))

    start_index = peak_index - (window_size // 2)
    start_index = max(0, start_index)
    start_index = min(start_index, len(candidates) - window_size)
    end_index = start_index + window_size

    return candidates[start_index:end_index]


def resolve_output_root(input_path: Path, output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root
    return input_path.parent / f"{input_path.name}_keyFrames"


def copy_patient_metadata_files(input_path: Path, output_root: Path, input_is_image_folder: bool) -> None:
    if input_is_image_folder:
        return

    for metadata_filename in PATIENT_METADATA_FILENAMES:
        for metadata_path in input_path.glob(f"*/{metadata_filename}"):
            relative_metadata_path = metadata_path.relative_to(input_path)
            destination_path = output_root / relative_metadata_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(metadata_path, destination_path)


def copy_patient_views_json(input_path: Path, output_root: Path, input_is_image_folder: bool) -> None:
    copy_patient_metadata_files(input_path, output_root, input_is_image_folder)


def resolve_sequence_output_dir(
    frames_dir: Path,
    input_path: Path,
    output_root: Path,
    input_is_image_folder: bool,
    frames_dirname: str,
) -> Path:
    if input_is_image_folder:
        return output_root

    if frames_dir.name == frames_dirname:
        relative_parent = frames_dir.parent.relative_to(input_path)
        return output_root / relative_parent

    relative_sequence_dir = frames_dir.relative_to(input_path)
    return output_root / relative_sequence_dir


def write_keyframes(
    candidates: list[KeyframeCandidate],
    output_dir: Path,
    overwrite: bool,
) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for candidate in candidates:
        write_grayscale_image(load_grayscale_image(candidate.source_path), output_dir / candidate.name)


def _extract_keyframes_job(
    job: ExtractionJob,
) -> ExtractionResult:
    (
        backend_name,
        frames_dir,
        output_dir,
        limit,
        baseline_frames,
        smoothing_window,
        overwrite,
        skip_existing,
    ) = job
    return extract_keyframes_from_directory(
        frames_dir=frames_dir,
        output_dir=output_dir,
        limit=limit,
        baseline_frames=baseline_frames,
        smoothing_window=smoothing_window,
        backend=backend_name,
        overwrite=overwrite,
        skip_existing=skip_existing,
    )


def _extract_keyframes_with_backend(
    frames_dir: Path,
    output_dir: Path,
    scoring_backend: ScoringBackend,
    limit: int = 6,
    baseline_frames: int = 3,
    smoothing_window: int = 5,
    overwrite: bool = False,
) -> ExtractionResult:
    candidates = scoring_backend.score_frame_directory(frames_dir, baseline_frames=baseline_frames)
    if not candidates:
        raise ValueError(f"No supported image files found in: {frames_dir}")

    selected = select_keyframe_window(candidates, limit, smoothing_window)
    write_keyframes(selected, output_dir, overwrite)

    return ExtractionResult(
        frames_dir=frames_dir,
        output_dir=output_dir,
        selected_count=len(selected),
    )


def build_skipped_result(frames_dir: Path, output_dir: Path) -> ExtractionResult:
    if not output_dir.is_dir():
        raise FileExistsError(f"Output path already exists and is not a directory: {output_dir}")

    return ExtractionResult(
        frames_dir=frames_dir,
        output_dir=output_dir,
        selected_count=len(list_image_files(output_dir)),
        skipped=True,
    )


def extract_keyframes_from_root(
    input_path: Path,
    output_root: Path | None = None,
    limit: int = 6,
    baseline_frames: int = 3,
    smoothing_window: int = 5,
    workers: int = 1,
    backend: BackendName = "cpu",
    frames_dirname: str = "frames",
    overwrite: bool = False,
    skip_existing: bool = False,
    cadica_selected_frame_counts: bool = False,
) -> list[ExtractionResult]:
    if workers <= 0:
        raise ValueError("workers must be greater than 0.")
    if backend not in BACKEND_CHOICES:
        raise ValueError(f"Unsupported backend: {backend}")
    if overwrite and skip_existing:
        raise ValueError("overwrite and skip_existing cannot both be enabled.")
    if backend != "cpu" and workers != 1:
        raise ValueError("workers > 1 is only supported with the cpu backend.")

    effective_frames_dirname = "input" if cadica_selected_frame_counts and frames_dirname == "frames" else frames_dirname
    discovery_input_path = (
        resolve_cadica_selected_videos_root(input_path)
        if cadica_selected_frame_counts
        else input_path
    )

    frame_directories = discover_frame_directories(discovery_input_path, effective_frames_dirname)
    if not frame_directories:
        raise ValueError(f"No supported image directories were found under: {discovery_input_path}")

    input_is_image_folder = bool(list_image_files(discovery_input_path))
    resolved_output_root = resolve_output_root(input_path, output_root)
    copy_patient_metadata_files(discovery_input_path, resolved_output_root, input_is_image_folder)
    jobs: list[ExtractionJob] = [
        (
            backend,
            frames_dir,
            resolve_sequence_output_dir(
                frames_dir=frames_dir,
                input_path=discovery_input_path,
                output_root=resolved_output_root,
                input_is_image_folder=input_is_image_folder,
                frames_dirname=effective_frames_dirname,
            ),
            _resolve_sequence_limit(
                frames_dir=frames_dir,
                default_limit=limit,
                frames_dirname=effective_frames_dirname,
                cadica_selected_frame_counts=cadica_selected_frame_counts,
            ),
            baseline_frames,
            smoothing_window,
            overwrite,
            skip_existing,
        )
        for frames_dir in frame_directories
    ]

    if not overwrite and not skip_existing:
        for _, _, output_dir, _, _, _, _, _ in jobs:
            if output_dir.exists():
                raise FileExistsError(
                    f"Output directory already exists: {output_dir}. "
                    "Use --overwrite to replace it."
                )

    results: list[ExtractionResult | None] = [None] * len(jobs)
    pending_jobs: list[tuple[int, ExtractionJob]] = []

    with tqdm(
        total=len(jobs),
        desc="Sequences",
        unit="seq",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    ) as progress:
        for index, job in enumerate(jobs):
            _, frames_dir, output_dir, _, _, _, _, job_skip_existing = job
            if job_skip_existing and output_dir.exists():
                results[index] = build_skipped_result(frames_dir, output_dir)
                progress.update(1)
                continue

            pending_jobs.append((index, job))

        if workers == 1 or len(pending_jobs) <= 1:
            scoring_backend: ScoringBackend | None = None
            for index, job in pending_jobs:
                (
                    _backend_name,
                    frames_dir,
                    output_dir,
                    job_limit,
                    job_baseline_frames,
                    job_smoothing_window,
                    job_overwrite,
                    _job_skip_existing,
                ) = job
                if scoring_backend is None:
                    scoring_backend = create_backend(backend)

                results[index] = _extract_keyframes_with_backend(
                    frames_dir=frames_dir,
                    output_dir=output_dir,
                    scoring_backend=scoring_backend,
                    limit=job_limit,
                    baseline_frames=job_baseline_frames,
                    smoothing_window=job_smoothing_window,
                    overwrite=job_overwrite,
                )
                progress.update(1)
        elif pending_jobs:
            max_workers = min(workers, len(pending_jobs))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(_extract_keyframes_job, job): index
                    for index, job in pending_jobs
                }
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    results[index] = future.result()
                    progress.update(1)

    return [result for result in results if result is not None]


def extract_keyframes_from_directory(
    frames_dir: Path,
    output_dir: Path,
    limit: int = 6,
    baseline_frames: int = 3,
    smoothing_window: int = 5,
    backend: BackendName = "cpu",
    overwrite: bool = False,
    skip_existing: bool = False,
) -> ExtractionResult:
    if backend not in BACKEND_CHOICES:
        raise ValueError(f"Unsupported backend: {backend}")
    if overwrite and skip_existing:
        raise ValueError("overwrite and skip_existing cannot both be enabled.")
    if skip_existing and output_dir.exists():
        return build_skipped_result(frames_dir, output_dir)

    return _extract_keyframes_with_backend(
        frames_dir=frames_dir,
        output_dir=output_dir,
        scoring_backend=create_backend(backend),
        limit=limit,
        baseline_frames=baseline_frames,
        smoothing_window=smoothing_window,
        overwrite=overwrite,
    )


def _resolve_sequence_limit(
    *,
    frames_dir: Path,
    default_limit: int,
    frames_dirname: str,
    cadica_selected_frame_counts: bool,
) -> int:
    if not cadica_selected_frame_counts:
        return default_limit
    return read_cadica_selected_frame_count(frames_dir, frames_dirname)
