from __future__ import annotations

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from .pipeline import PipelineConfig, run_stenosis_detection, run_stenosis_detection_variants
from .visualization import build_output_paths, save_detection_outputs

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


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ORIGINAL_SLICE_PATTERN = re.compile(r"^slice_(\d+)$", re.IGNORECASE)
MASK_SLICE_PATTERN = re.compile(r"^slice_(\d+)_mask$", re.IGNORECASE)


@dataclass(slots=True)
class TreeProcessingJob:
    image_path: Path
    mask_path: Path
    relative_dir: Path
    image_stem: str


@dataclass(slots=True)
class BatchFailure:
    image_path: str
    mask_path: str
    error: str


@dataclass(slots=True)
class BatchProcessSummary:
    images_root: Path
    masks_root: Path
    output_root: Path
    total_jobs: int
    workers: int
    processed: int
    skipped_existing: int
    failed: int
    failures: list[BatchFailure]
    threshold_variants: list[str] | None = None
    debug_images_saved: bool = True

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "images_root": str(self.images_root),
            "masks_root": str(self.masks_root),
            "output_root": str(self.output_root),
            "total_jobs": self.total_jobs,
            "workers": self.workers,
            "processed": self.processed,
            "skipped_existing": self.skipped_existing,
            "failed": self.failed,
            "failures": [asdict(failure) for failure in self.failures],
            "debug_images_saved": self.debug_images_saved,
        }
        if self.threshold_variants is not None:
            payload["threshold_variants"] = list(self.threshold_variants)
        return payload


def discover_tree_jobs(images_root: str | Path, masks_root: str | Path) -> list[TreeProcessingJob]:
    resolved_images_root = Path(images_root)
    resolved_masks_root = Path(masks_root)

    if not resolved_images_root.is_dir():
        raise NotADirectoryError(f"Images root does not exist or is not a directory: {resolved_images_root}")
    if not resolved_masks_root.is_dir():
        raise NotADirectoryError(f"Masks root does not exist or is not a directory: {resolved_masks_root}")

    mask_index = _build_mask_index(resolved_masks_root)
    jobs: list[TreeProcessingJob] = []
    missing_masks: list[str] = []

    for image_path in sorted(_iter_candidate_images(resolved_images_root)):
        relative_path = image_path.relative_to(resolved_images_root)
        mask_key = (relative_path.parent.as_posix(), f"{image_path.stem}_mask")
        mask_path = mask_index.get(mask_key)

        if mask_path is None:
            missing_masks.append(str(relative_path))
            continue

        jobs.append(
            TreeProcessingJob(
                image_path=image_path,
                mask_path=mask_path,
                relative_dir=relative_path.parent,
                image_stem=image_path.stem,
            )
        )

    if missing_masks:
        preview = ", ".join(missing_masks[:10])
        suffix = " ..." if len(missing_masks) > 10 else ""
        raise FileNotFoundError(
            f"Missing mask files for {len(missing_masks)} image(s). "
            f"Expected matching slice_*_mask files in the mirrored mask tree. "
            f"Examples: {preview}{suffix}"
        )

    if not jobs:
        raise FileNotFoundError(f"No slice_#### image files were found under: {resolved_images_root}")

    return jobs


def process_tree(
    jobs: list[TreeProcessingJob],
    output_root: str | Path,
    *,
    images_root: str | Path,
    masks_root: str | Path,
    config: PipelineConfig | None = None,
    skip_existing: bool = True,
    workers: int = 1,
    threshold_variants: list[tuple[str, PipelineConfig]] | None = None,
    write_debug_images: bool = True,
) -> BatchProcessSummary:
    pipeline_config = config or PipelineConfig()
    worker_count = _resolve_worker_count(workers)
    resolved_output_root = Path(output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    pending_jobs = _filter_pending_jobs(
        jobs,
        resolved_output_root,
        skip_existing,
        threshold_variants,
        write_debug_images,
    )
    skipped_existing = len(jobs) - len(pending_jobs)
    failures: list[BatchFailure] = []

    with tqdm(total=len(jobs), desc="Processing slices", unit="slice", dynamic_ncols=True) as progress:
        if skipped_existing:
            progress.update(skipped_existing)
            progress.set_postfix(processed=processed, skipped=skipped_existing, failed=len(failures))

        if worker_count == 1:
            for job in pending_jobs:
                failure = _process_tree_job(
                    job,
                    resolved_output_root,
                    pipeline_config,
                    threshold_variants,
                    write_debug_images,
                )
                if failure is None:
                    processed += 1
                else:
                    failures.append(failure)
                    progress.write(f"Failed: {job.image_path} -> {failure.error}")

                progress.update(1)
                progress.set_postfix(processed=processed, skipped=skipped_existing, failed=len(failures))
        elif pending_jobs:
            future_to_job = {}
            with ProcessPoolExecutor(max_workers=worker_count, initializer=_prepare_worker_process) as executor:
                for job in pending_jobs:
                    future = executor.submit(
                        _process_tree_job,
                        job,
                        resolved_output_root,
                        pipeline_config,
                        threshold_variants,
                        write_debug_images,
                    )
                    future_to_job[future] = job

                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        failure = future.result()
                    except Exception as exc:  # pragma: no cover - protects parent process progress.
                        failure = BatchFailure(
                            image_path=str(job.image_path),
                            mask_path=str(job.mask_path),
                            error=str(exc),
                        )

                    if failure is None:
                        processed += 1
                    else:
                        failures.append(failure)
                        progress.write(f"Failed: {job.image_path} -> {failure.error}")

                    progress.update(1)
                    progress.set_postfix(processed=processed, skipped=skipped_existing, failed=len(failures))

    summary = BatchProcessSummary(
        images_root=Path(images_root),
        masks_root=Path(masks_root),
        output_root=resolved_output_root,
        total_jobs=len(jobs),
        workers=worker_count,
        processed=processed,
        skipped_existing=skipped_existing,
        failed=len(failures),
        failures=failures,
        threshold_variants=None if threshold_variants is None else [name for name, _ in threshold_variants],
        debug_images_saved=write_debug_images,
    )

    (resolved_output_root / "batch_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2),
        encoding="utf-8",
    )
    return summary


def _iter_candidate_images(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        if ORIGINAL_SLICE_PATTERN.match(path.stem):
            yield path


def _build_mask_index(masks_root: Path) -> dict[tuple[str, str], Path]:
    mask_index: dict[tuple[str, str], Path] = {}

    for path in masks_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        if not MASK_SLICE_PATTERN.match(path.stem):
            continue

        relative_path = path.relative_to(masks_root)
        key = (relative_path.parent.as_posix(), path.stem)

        if key in mask_index:
            raise FileExistsError(
                f"Duplicate mask candidates found for {relative_path.parent / path.stem}: "
                f"{mask_index[key]} and {path}"
            )

        mask_index[key] = path

    return mask_index


def _resolve_worker_count(workers: int) -> int:
    if workers < 0:
        raise ValueError("workers must be 0 or greater")
    if workers == 0:
        return max(1, os.cpu_count() or 1)
    return workers


def _filter_pending_jobs(
    jobs: list[TreeProcessingJob],
    output_root: Path,
    skip_existing: bool,
    threshold_variants: list[tuple[str, PipelineConfig]] | None,
    write_debug_images: bool,
) -> list[TreeProcessingJob]:
    if not skip_existing:
        return jobs

    pending_jobs: list[TreeProcessingJob] = []
    for job in jobs:
        expected_outputs = _build_expected_job_outputs(job, output_root, threshold_variants, write_debug_images)
        if not all(path.exists() for path in expected_outputs):
            pending_jobs.append(job)
    return pending_jobs


def _build_expected_job_outputs(
    job: TreeProcessingJob,
    output_root: Path,
    threshold_variants: list[tuple[str, PipelineConfig]] | None,
    write_debug_images: bool,
) -> list[Path]:
    if threshold_variants is None:
        output_dir = output_root / job.relative_dir
        return _expected_output_paths(output_dir, job.image_stem, write_debug_images)

    expected_outputs: list[Path] = []
    for variant_name, _ in threshold_variants:
        output_dir = output_root / variant_name / job.relative_dir
        expected_outputs.extend(_expected_output_paths(output_dir, job.image_stem, write_debug_images))
    return expected_outputs


def _expected_output_paths(output_dir: Path, file_prefix: str, write_debug_images: bool) -> list[Path]:
    output_paths = build_output_paths(output_dir, file_prefix=file_prefix)
    if not write_debug_images:
        return [output_paths["results_json"]]
    return list(output_paths.values())


def _prepare_worker_process() -> None:
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass


def _process_tree_job(
    job: TreeProcessingJob,
    output_root: Path,
    config: PipelineConfig,
    threshold_variants: list[tuple[str, PipelineConfig]] | None = None,
    write_debug_images: bool = True,
) -> BatchFailure | None:
    try:
        if threshold_variants is None:
            output_dir = output_root / job.relative_dir
            result = run_stenosis_detection(job.image_path, job.mask_path, config=config)
            save_detection_outputs(
                result,
                output_dir,
                file_prefix=job.image_stem,
                show=False,
                write_debug_images=write_debug_images,
            )
        else:
            variant_configs = [variant_config for _, variant_config in threshold_variants]
            results = run_stenosis_detection_variants(job.image_path, job.mask_path, variant_configs)
            for (variant_name, _), result in zip(threshold_variants, results, strict=True):
                output_dir = output_root / variant_name / job.relative_dir
                save_detection_outputs(
                    result,
                    output_dir,
                    file_prefix=job.image_stem,
                    show=False,
                    write_debug_images=write_debug_images,
                )
    except Exception as exc:  # pragma: no cover - depends on input data.
        return BatchFailure(
            image_path=str(job.image_path),
            mask_path=str(job.mask_path),
            error=str(exc),
        )
    return None
