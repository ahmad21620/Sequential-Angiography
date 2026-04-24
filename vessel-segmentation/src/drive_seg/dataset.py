from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augmentations import PatchAugmenter
from .preprocessing import binarize_masks, preprocess_image
from .utils import list_image_files

SPLIT_IMAGE_DIRNAME = "Original"
SPLIT_MASK_DIRNAME = "Mask"
SPLIT_FOV_DIRNAME = "FOV"

FOVMode = Literal["provided", "auto_generated"]


class DatasetValidationError(ValueError):
    """Raised when a dataset split does not match the required on-disk layout."""


@dataclass(frozen=True)
class SegmentationSamplePaths:
    image_id: str
    image_path: Path
    mask_path: Path
    fov_path: Path | None


@dataclass
class SegmentationSplit:
    split_name: str
    samples: list[SegmentationSamplePaths]
    image_ids: list[str]
    fov_mode: FOVMode


@dataclass(frozen=True)
class DatasetSplitReport:
    split_name: str
    original_count: int
    mask_count: int
    fov_count: int
    paired_count: int
    fov_mode: FOVMode


@dataclass(frozen=True)
class DatasetValidationReport:
    dataset_root: Path
    split_reports: list[DatasetSplitReport]
    issues: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class _DirectoryIndex:
    file_count: int
    paths_by_stem: dict[str, Path]
    issues: list[str]


def _load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.transpose(array, (2, 0, 1))


def _load_single_channel_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    return array[np.newaxis, ...]


def _load_image_preserve_channels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image, dtype=np.uint8)

    if array.ndim == 2:
        return array[np.newaxis, ...]
    if array.ndim == 3:
        if array.shape[2] == 1:
            return np.transpose(array, (2, 0, 1))
        if array.shape[2] >= 3:
            return np.transpose(array[..., :3], (2, 0, 1))

    raise ValueError(f"Unsupported image shape {array.shape} for {path}")


def _read_spatial_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return (height, width)


def _format_values(values: Sequence[str], limit: int = 20) -> str:
    ordered = list(values)
    if len(ordered) <= limit:
        return ", ".join(ordered)
    preview = ", ".join(ordered[:limit])
    return f"{preview}, ... (+{len(ordered) - limit} more)"


def _index_files_by_stem(
    directory: Path,
    split_name: str,
    label: str,
) -> _DirectoryIndex:
    issues: list[str] = []
    if not directory.exists():
        issues.append(
            f"Split '{split_name}' is missing the {label} directory: {directory}"
        )
        return _DirectoryIndex(file_count=0, paths_by_stem={}, issues=issues)
    if not directory.is_dir():
        issues.append(
            f"Split '{split_name}' expected {directory} to be a directory for {label} files."
        )
        return _DirectoryIndex(file_count=0, paths_by_stem={}, issues=issues)

    try:
        file_paths = list_image_files(directory)
    except FileNotFoundError:
        issues.append(
            f"Split '{split_name}' has no supported image files in {directory}."
        )
        return _DirectoryIndex(file_count=0, paths_by_stem={}, issues=issues)

    grouped_paths: dict[str, list[Path]] = {}
    for file_path in file_paths:
        grouped_paths.setdefault(file_path.stem, []).append(file_path)

    unique_paths: dict[str, Path] = {}
    duplicate_stems: list[str] = []
    for stem, paths in sorted(grouped_paths.items()):
        if len(paths) > 1:
            duplicate_stems.append(
                f"{stem} -> {', '.join(path.name for path in paths)}"
            )
            continue
        unique_paths[stem] = paths[0]

    if duplicate_stems:
        issues.append(
            f"Split '{split_name}' has duplicate {label} stems: {_format_values(duplicate_stems)}"
        )

    return _DirectoryIndex(
        file_count=len(file_paths),
        paths_by_stem=unique_paths,
        issues=issues,
    )


def _empty_directory_index() -> _DirectoryIndex:
    return _DirectoryIndex(file_count=0, paths_by_stem={}, issues=[])


def _determine_fov_mode(split_dir: Path) -> FOVMode:
    return "provided" if (split_dir / SPLIT_FOV_DIRNAME).exists() else "auto_generated"


def _inspect_split(
    dataset_root: Path,
    split_name: str,
) -> tuple[DatasetSplitReport, list[SegmentationSamplePaths], list[str]]:
    split_dir = dataset_root / split_name
    original_index = _index_files_by_stem(
        split_dir / SPLIT_IMAGE_DIRNAME,
        split_name=split_name,
        label=SPLIT_IMAGE_DIRNAME,
    )
    mask_index = _index_files_by_stem(
        split_dir / SPLIT_MASK_DIRNAME,
        split_name=split_name,
        label=SPLIT_MASK_DIRNAME,
    )
    fov_mode = _determine_fov_mode(split_dir)
    if fov_mode == "provided":
        fov_index = _index_files_by_stem(
            split_dir / SPLIT_FOV_DIRNAME,
            split_name=split_name,
            label=SPLIT_FOV_DIRNAME,
        )
    else:
        fov_index = _empty_directory_index()

    issues = [
        *original_index.issues,
        *mask_index.issues,
        *fov_index.issues,
    ]

    original_stems = set(original_index.paths_by_stem)
    mask_stems = set(mask_index.paths_by_stem)
    paired_image_mask_stems = original_stems & mask_stems

    missing_from_mask = sorted(original_stems - mask_stems)
    missing_from_original = sorted(mask_stems - original_stems)
    if missing_from_mask:
        issues.append(
            f"Split '{split_name}' is missing Mask files for stems: {_format_values(missing_from_mask)}"
        )
    if missing_from_original:
        issues.append(
            "Split "
            f"'{split_name}' has Mask files without matching Original files for stems: "
            f"{_format_values(missing_from_original)}"
        )

    if fov_mode == "provided":
        fov_stems = set(fov_index.paths_by_stem)
        missing_from_fov = sorted(paired_image_mask_stems - fov_stems)
        missing_from_image_mask_pair = sorted(fov_stems - paired_image_mask_stems)
        if missing_from_fov:
            issues.append(
                f"Split '{split_name}' is missing FOV files for stems: {_format_values(missing_from_fov)}"
            )
        if missing_from_image_mask_pair:
            issues.append(
                "Split "
                f"'{split_name}' has FOV files without matching Original/Mask pairs for stems: "
                f"{_format_values(missing_from_image_mask_pair)}"
            )
        paired_stems = sorted(paired_image_mask_stems & fov_stems)
    else:
        paired_stems = sorted(paired_image_mask_stems)

    sample_paths: list[SegmentationSamplePaths] = []
    size_mismatches: list[str] = []
    for stem in paired_stems:
        image_path = original_index.paths_by_stem[stem]
        mask_path = mask_index.paths_by_stem[stem]

        image_size = _read_spatial_size(image_path)
        mask_size = _read_spatial_size(mask_path)
        if image_size != mask_size:
            size_mismatches.append(
                f"{stem} -> Original{image_size}, Mask{mask_size}"
            )
            continue

        fov_path: Path | None = None
        if fov_mode == "provided":
            fov_path = fov_index.paths_by_stem[stem]
            fov_size = _read_spatial_size(fov_path)
            if image_size != fov_size:
                size_mismatches.append(
                    f"{stem} -> Original{image_size}, Mask{mask_size}, FOV{fov_size}"
                )
                continue

        sample_paths.append(
            SegmentationSamplePaths(
                image_id=stem,
                image_path=image_path,
                mask_path=mask_path,
                fov_path=fov_path,
            )
        )

    if size_mismatches:
        issues.append(
            f"Split '{split_name}' has size mismatches: {_format_values(size_mismatches)}"
        )

    report = DatasetSplitReport(
        split_name=split_name,
        original_count=original_index.file_count,
        mask_count=mask_index.file_count,
        fov_count=fov_index.file_count,
        paired_count=len(sample_paths),
        fov_mode=fov_mode,
    )
    return report, sample_paths, issues


def _format_split_issues(
    dataset_root: Path,
    split_name: str,
    issues: Sequence[str],
) -> str:
    lines = [
        f"Dataset split '{split_name}' under {dataset_root} is invalid:",
    ]
    lines.extend(f"- {issue}" for issue in issues)
    return "\n".join(lines)


def validate_dataset_root(
    dataset_root: Path,
    splits: Sequence[str] = ("train", "val", "test"),
) -> DatasetValidationReport:
    split_reports: list[DatasetSplitReport] = []
    issues: list[str] = []
    for split_name in splits:
        report, _, split_issues = _inspect_split(dataset_root, split_name)
        split_reports.append(report)
        issues.extend(split_issues)
    return DatasetValidationReport(
        dataset_root=dataset_root,
        split_reports=split_reports,
        issues=issues,
    )


def _format_fov_status(split_report: DatasetSplitReport) -> str:
    if split_report.fov_mode == "provided":
        return f"provided({split_report.fov_count})"
    return "auto-generated(full-image)"


def format_validation_report(report: DatasetValidationReport) -> str:
    lines = [f"Dataset root: {report.dataset_root}", "Split counts:"]
    for split_report in report.split_reports:
        lines.append(
            f"- {split_report.split_name}: "
            f"Original={split_report.original_count}, "
            f"Mask={split_report.mask_count}, "
            f"FOV={_format_fov_status(split_report)}, "
            f"paired={split_report.paired_count}"
        )
    if report.issues:
        lines.append("Problems:")
        lines.extend(f"- {issue}" for issue in report.issues)
    else:
        lines.append("Validation passed.")
    return "\n".join(lines)


def load_dataset_split(dataset_root: Path, split_name: str) -> SegmentationSplit:
    report, sample_paths, issues = _inspect_split(dataset_root, split_name)
    if issues:
        raise DatasetValidationError(
            _format_split_issues(dataset_root, split_name, issues)
        )
    if report.paired_count == 0:
        raise DatasetValidationError(
            f"Dataset split '{split_name}' under {dataset_root} contains no valid image/mask pairs."
        )

    return SegmentationSplit(
        split_name=split_name,
        samples=sample_paths,
        image_ids=[sample.image_id for sample in sample_paths],
        fov_mode=report.fov_mode,
    )


def load_rgb_image(path: Path) -> np.ndarray:
    return _load_rgb_image(path)


def load_sample_image(sample: SegmentationSamplePaths) -> np.ndarray:
    return _load_image_preserve_channels(sample.image_path)


def load_sample_mask(sample: SegmentationSamplePaths) -> np.ndarray:
    return _load_single_channel_image(sample.mask_path)


def load_sample_fov_mask(
    sample: SegmentationSamplePaths,
    spatial_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if sample.fov_path is None:
        if spatial_shape is None:
            spatial_shape = _read_spatial_size(sample.mask_path)
        return np.ones((1, spatial_shape[0], spatial_shape[1]), dtype=np.uint8)
    return _load_single_channel_image(sample.fov_path)


def resolve_custom_image_paths(
    input_path: Path,
    ignore_names: set[str] | None = None,
) -> tuple[list[Path], list[Path]]:
    image_paths = list_image_files(input_path, recursive=input_path.is_dir())
    ignored_names = {name.lower() for name in ignore_names or set()}
    if ignored_names:
        image_paths = [path for path in image_paths if path.name.lower() not in ignored_names]
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {input_path} after filtering.")
    if input_path.is_dir():
        image_names = [path.relative_to(input_path) for path in image_paths]
    else:
        image_names = [Path(path.name) for path in image_paths]
    return image_names, image_paths


class RandomPatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        images: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        coordinates: np.ndarray,
        patch_size: tuple[int, int],
        augmenter: PatchAugmenter | None = None,
    ) -> None:
        self.images = [image.astype(np.float32, copy=False) for image in images]
        self.masks = [mask.astype(np.float32, copy=False) for mask in masks]
        self.coordinates = coordinates.astype(np.int32, copy=False)
        self.patch_h, self.patch_w = patch_size
        self.augmenter = augmenter

    def __len__(self) -> int:
        return int(self.coordinates.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_index, top, left = self.coordinates[index]
        bottom = top + self.patch_h
        right = left + self.patch_w

        image_patch = self.images[image_index][:, top:bottom, left:right].copy()
        mask_patch = self.masks[image_index][:, top:bottom, left:right].copy()
        if self.augmenter is not None:
            augmentation_seed = int(
                torch.randint(0, 2**31 - 1, size=(1,), dtype=torch.int64).item()
            )
            image_patch, mask_patch = self.augmenter(
                image_patch,
                mask_patch,
                seed=augmentation_seed,
            )

        return torch.from_numpy(image_patch), torch.from_numpy(mask_patch)


class LazyRandomPatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        samples: Sequence[SegmentationSamplePaths],
        coordinates: np.ndarray,
        patch_size: tuple[int, int],
        augmenter: PatchAugmenter | None = None,
        max_cached_images: int = 2,
    ) -> None:
        self.samples = list(samples)
        self.coordinates = coordinates.astype(np.int32, copy=False)
        self.patch_h, self.patch_w = patch_size
        self.augmenter = augmenter
        self.max_cached_images = max_cached_images
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return int(self.coordinates.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_index, top, left = self.coordinates[index]
        bottom = top + self.patch_h
        right = left + self.patch_w

        image, mask = self._get_cached_image_and_mask(int(image_index))
        image_patch = image[:, top:bottom, left:right].copy()
        mask_patch = mask[:, top:bottom, left:right].copy()
        if self.augmenter is not None:
            augmentation_seed = int(
                torch.randint(0, 2**31 - 1, size=(1,), dtype=torch.int64).item()
            )
            image_patch, mask_patch = self.augmenter(
                image_patch,
                mask_patch,
                seed=augmentation_seed,
            )

        return torch.from_numpy(image_patch), torch.from_numpy(mask_patch)

    def _get_cached_image_and_mask(self, image_index: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cache.get(image_index)
        if cached is not None:
            self._cache.move_to_end(image_index)
            return cached

        sample = self.samples[image_index]
        image = preprocess_image(load_sample_image(sample)).astype(np.float32, copy=False)
        mask = binarize_masks(load_sample_mask(sample)).astype(np.float32, copy=False)

        cached = (image, mask)
        self._cache[image_index] = cached
        if self.max_cached_images > 0 and len(self._cache) > self.max_cached_images:
            self._cache.popitem(last=False)
        return cached


class PatchArrayDataset(Dataset[torch.Tensor]):
    def __init__(self, patches: np.ndarray) -> None:
        self.patches = patches.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(self.patches.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.patches[index].copy())
