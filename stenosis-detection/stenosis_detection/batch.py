from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .pipeline import PipelineConfig, run_stenosis_detection
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
    processed: int
    skipped_existing: int
    failed: int
    failures: list[BatchFailure]

    def to_dict(self) -> dict[str, object]:
        return {
            "images_root": str(self.images_root),
            "masks_root": str(self.masks_root),
            "output_root": str(self.output_root),
            "total_jobs": self.total_jobs,
            "processed": self.processed,
            "skipped_existing": self.skipped_existing,
            "failed": self.failed,
            "failures": [asdict(failure) for failure in self.failures],
        }


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
) -> BatchProcessSummary:
    pipeline_config = config or PipelineConfig()
    resolved_output_root = Path(output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped_existing = 0
    failures: list[BatchFailure] = []

    with tqdm(total=len(jobs), desc="Processing slices", unit="slice", dynamic_ncols=True) as progress:
        for job in jobs:
            output_dir = resolved_output_root / job.relative_dir
            expected_outputs = build_output_paths(output_dir, file_prefix=job.image_stem)

            if skip_existing and all(path.exists() for path in expected_outputs.values()):
                skipped_existing += 1
                progress.update(1)
                progress.set_postfix(processed=processed, skipped=skipped_existing, failed=len(failures))
                continue

            try:
                result = run_stenosis_detection(job.image_path, job.mask_path, config=pipeline_config)
                save_detection_outputs(result, output_dir, file_prefix=job.image_stem, show=False)
                processed += 1
            except Exception as exc:  # pragma: no cover - depends on input data.
                failures.append(
                    BatchFailure(
                        image_path=str(job.image_path),
                        mask_path=str(job.mask_path),
                        error=str(exc),
                    )
                )
                progress.write(f"Failed: {job.image_path} -> {exc}")

            progress.update(1)
            progress.set_postfix(processed=processed, skipped=skipped_existing, failed=len(failures))

    summary = BatchProcessSummary(
        images_root=Path(images_root),
        masks_root=Path(masks_root),
        output_root=resolved_output_root,
        total_jobs=len(jobs),
        processed=processed,
        skipped_existing=skipped_existing,
        failed=len(failures),
        failures=failures,
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
