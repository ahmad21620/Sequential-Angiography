from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F


PATCH_CATEGORY_ORDER = ("vessel", "hard_negative", "random_fov")


@dataclass(frozen=True)
class PatchSamplingResult:
    coordinates: np.ndarray
    category_counts: dict[str, int]


MaskPairLoader = Callable[[int], tuple[np.ndarray, np.ndarray | None]]


def sample_patch_coordinates(
    masks: Sequence[np.ndarray],
    fov_masks: Sequence[np.ndarray] | None,
    patch_size: tuple[int, int],
    num_patches: int,
    seed: int,
    vessel_patch_ratio: float,
    hard_negative_patch_ratio: float,
    min_fov_coverage: float,
    hard_negative_band_width: int,
) -> PatchSamplingResult:
    image_count = len(masks)
    if image_count == 0:
        raise ValueError("At least one image mask is required for patch sampling.")
    if fov_masks is not None and len(fov_masks) != image_count:
        raise ValueError("fov_masks must have the same length as masks.")

    def load_mask_pair(image_index: int) -> tuple[np.ndarray, np.ndarray | None]:
        image_fov_mask = fov_masks[image_index] if fov_masks is not None else None
        return masks[image_index], image_fov_mask

    return sample_patch_coordinates_streaming(
        image_count=image_count,
        load_mask_pair=load_mask_pair,
        patch_size=patch_size,
        num_patches=num_patches,
        seed=seed,
        vessel_patch_ratio=vessel_patch_ratio,
        hard_negative_patch_ratio=hard_negative_patch_ratio,
        min_fov_coverage=min_fov_coverage,
        hard_negative_band_width=hard_negative_band_width,
    )


def sample_patch_coordinates_streaming(
    image_count: int,
    load_mask_pair: MaskPairLoader,
    patch_size: tuple[int, int],
    num_patches: int,
    seed: int,
    vessel_patch_ratio: float,
    hard_negative_patch_ratio: float,
    min_fov_coverage: float,
    hard_negative_band_width: int,
) -> PatchSamplingResult:
    if image_count == 0:
        raise ValueError("At least one image mask is required for patch sampling.")

    _validate_sampling_parameters(
        num_patches=num_patches,
        vessel_patch_ratio=vessel_patch_ratio,
        hard_negative_patch_ratio=hard_negative_patch_ratio,
        min_fov_coverage=min_fov_coverage,
        hard_negative_band_width=hard_negative_band_width,
    )

    category_image_counts = {
        category: np.zeros(image_count, dtype=np.int64) for category in PATCH_CATEGORY_ORDER
    }
    for image_index in range(image_count):
        mask, image_fov_mask = load_mask_pair(image_index)
        image_candidates = _build_patch_candidate_masks(
            mask=mask,
            fov_mask=image_fov_mask,
            patch_size=patch_size,
            min_fov_coverage=min_fov_coverage,
            hard_negative_band_width=hard_negative_band_width,
        )
        for category in PATCH_CATEGORY_ORDER:
            category_image_counts[category][image_index] = int(image_candidates[category].sum())

    available_categories = [
        category
        for category in PATCH_CATEGORY_ORDER
        if int(category_image_counts[category].sum()) > 0
    ]
    if not available_categories:
        raise ValueError("No valid patch candidates found for the requested patch size and FOV rules.")

    base_ratios = {
        "vessel": vessel_patch_ratio,
        "hard_negative": hard_negative_patch_ratio,
        "random_fov": 1.0 - vessel_patch_ratio - hard_negative_patch_ratio,
    }
    active_ratios = _resolve_active_ratios(base_ratios, available_categories)
    category_sample_counts = _allocate_counts(num_patches, active_ratios)

    rng = np.random.default_rng(seed)
    category_image_sample_counts = {
        category: _allocate_image_sample_counts(
            image_candidate_counts=category_image_counts[category],
            num_samples=category_sample_counts.get(category, 0),
            rng=rng,
        )
        for category in PATCH_CATEGORY_ORDER
    }

    coordinate_batches: list[np.ndarray] = []
    actual_category_counts = {category: 0 for category in PATCH_CATEGORY_ORDER}
    for category in PATCH_CATEGORY_ORDER:
        sample_count = category_sample_counts.get(category, 0)
        if sample_count == 0:
            continue

        sampled_coordinates = _sample_category_coordinates_streaming(
            image_count=image_count,
            load_mask_pair=load_mask_pair,
            patch_size=patch_size,
            min_fov_coverage=min_fov_coverage,
            hard_negative_band_width=hard_negative_band_width,
            category=category,
            image_sample_counts=category_image_sample_counts[category],
            rng=rng,
        )
        coordinate_batches.append(sampled_coordinates)
        actual_category_counts[category] = int(sampled_coordinates.shape[0])

    if not coordinate_batches:
        raise ValueError("Patch sampling produced no coordinates.")

    coordinates = np.concatenate(coordinate_batches, axis=0)
    if coordinates.shape[0] != num_patches:
        raise RuntimeError(
            f"Expected {num_patches} sampled patches, got {coordinates.shape[0]}."
        )

    shuffled_coordinates = coordinates[rng.permutation(coordinates.shape[0])]
    return PatchSamplingResult(
        coordinates=shuffled_coordinates.astype(np.int32, copy=False),
        category_counts=actual_category_counts,
    )


def _allocate_image_sample_counts(
    image_candidate_counts: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_samples == 0:
        return np.zeros_like(image_candidate_counts)

    total_candidates = int(image_candidate_counts.sum())
    if total_candidates == 0:
        raise ValueError("Cannot allocate samples for a patch category with no valid candidates.")

    image_weights = image_candidate_counts.astype(np.float64) / total_candidates
    return rng.multinomial(num_samples, image_weights)


def _validate_sampling_parameters(
    num_patches: int,
    vessel_patch_ratio: float,
    hard_negative_patch_ratio: float,
    min_fov_coverage: float,
    hard_negative_band_width: int,
) -> None:
    if num_patches <= 0:
        raise ValueError("num_patches must be greater than zero.")
    if not 0.0 <= vessel_patch_ratio <= 1.0:
        raise ValueError("vessel_patch_ratio must be in the range [0, 1].")
    if not 0.0 <= hard_negative_patch_ratio <= 1.0:
        raise ValueError("hard_negative_patch_ratio must be in the range [0, 1].")
    if vessel_patch_ratio + hard_negative_patch_ratio > 1.0:
        raise ValueError(
            "vessel_patch_ratio + hard_negative_patch_ratio must be less than or equal to 1."
        )
    if not 0.0 <= min_fov_coverage <= 1.0:
        raise ValueError("min_fov_coverage must be in the range [0, 1].")
    if hard_negative_band_width < 0:
        raise ValueError("hard_negative_band_width must be non-negative.")


def _resolve_active_ratios(
    base_ratios: dict[str, float],
    available_categories: Sequence[str],
) -> dict[str, float]:
    active_categories = [
        category
        for category in PATCH_CATEGORY_ORDER
        if category in available_categories and base_ratios[category] > 0.0
    ]
    if not active_categories:
        uniform_ratio = 1.0 / len(available_categories)
        return {category: uniform_ratio for category in available_categories}

    ratio_sum = sum(base_ratios[category] for category in active_categories)
    return {
        category: base_ratios[category] / ratio_sum for category in active_categories
    }


def _allocate_counts(total_count: int, ratios: dict[str, float]) -> dict[str, int]:
    raw_counts = {category: total_count * ratio for category, ratio in ratios.items()}
    counts = {category: int(np.floor(raw_count)) for category, raw_count in raw_counts.items()}
    remaining = total_count - sum(counts.values())

    ranked_categories = sorted(
        raw_counts,
        key=lambda category: (
            raw_counts[category] - counts[category],
            -PATCH_CATEGORY_ORDER.index(category),
        ),
        reverse=True,
    )
    for category in ranked_categories[:remaining]:
        counts[category] += 1
    return counts


def _sample_category_coordinates_streaming(
    image_count: int,
    load_mask_pair: MaskPairLoader,
    patch_size: tuple[int, int],
    min_fov_coverage: float,
    hard_negative_band_width: int,
    category: str,
    image_sample_counts: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    num_samples = int(image_sample_counts.sum())
    if num_samples == 0:
        return np.empty((0, 3), dtype=np.int32)

    coordinates = np.empty((num_samples, 3), dtype=np.int32)
    cursor = 0
    for image_index in range(image_count):
        sample_count = int(image_sample_counts[image_index])
        if sample_count == 0:
            continue

        mask, image_fov_mask = load_mask_pair(image_index)
        image_candidates = _build_patch_candidate_masks(
            mask=mask,
            fov_mask=image_fov_mask,
            patch_size=patch_size,
            min_fov_coverage=min_fov_coverage,
            hard_negative_band_width=hard_negative_band_width,
        )
        image_candidate_mask = image_candidates[category]
        candidate_indices = np.flatnonzero(image_candidate_mask)
        if candidate_indices.size == 0:
            raise RuntimeError(
                f"Patch candidates disappeared for category '{category}' on image index {image_index}."
            )
        sampled_indices = rng.choice(candidate_indices, size=sample_count, replace=True)
        rows, cols = np.divmod(sampled_indices, image_candidate_mask.shape[1])

        coordinates[cursor : cursor + sample_count, 0] = image_index
        coordinates[cursor : cursor + sample_count, 1] = rows.astype(np.int32, copy=False)
        coordinates[cursor : cursor + sample_count, 2] = cols.astype(np.int32, copy=False)
        cursor += sample_count

    if cursor != num_samples:
        raise RuntimeError(
            f"Expected {num_samples} sampled coordinates for category '{category}', got {cursor}."
        )
    return coordinates


def _build_patch_candidate_masks(
    mask: np.ndarray,
    fov_mask: np.ndarray | None,
    patch_size: tuple[int, int],
    min_fov_coverage: float,
    hard_negative_band_width: int,
) -> dict[str, np.ndarray]:
    vessel_mask = _to_binary_2d(mask)
    fov_mask_2d = _to_binary_2d(fov_mask) if fov_mask is not None else np.ones_like(vessel_mask)

    patch_h, patch_w = patch_size
    image_height, image_width = vessel_mask.shape
    if patch_h > image_height or patch_w > image_width:
        raise ValueError(
            f"Patch size {patch_size} cannot be larger than image shape {(image_height, image_width)}."
        )

    valid_patch_mask = _window_sum(fov_mask_2d.astype(np.float32, copy=False), patch_size) >= (
        min_fov_coverage * patch_h * patch_w
    )
    grid_height, grid_width = valid_patch_mask.shape
    center_row_offset = patch_h // 2
    center_col_offset = patch_w // 2
    center_rows = slice(center_row_offset, center_row_offset + grid_height)
    center_cols = slice(center_col_offset, center_col_offset + grid_width)

    center_fov_mask = fov_mask_2d[center_rows, center_cols]
    valid_center_mask = valid_patch_mask & center_fov_mask

    vessel_center_mask = vessel_mask[center_rows, center_cols]
    hard_negative_band = _build_hard_negative_band(vessel_mask, hard_negative_band_width)
    hard_negative_center_mask = hard_negative_band[center_rows, center_cols]

    vessel_candidates = valid_center_mask & vessel_center_mask
    hard_negative_candidates = valid_center_mask & hard_negative_center_mask
    random_fov_candidates = valid_center_mask & ~(vessel_candidates | hard_negative_candidates)

    return {
        "vessel": vessel_candidates,
        "hard_negative": hard_negative_candidates,
        "random_fov": random_fov_candidates,
    }


def _to_binary_2d(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"Expected a single-channel mask, got shape {array.shape}.")
        array = array[0]
    elif array.ndim != 2:
        raise ValueError(f"Expected a 2D or 3D mask, got shape {array.shape}.")
    return array > 0


def _window_sum(mask: np.ndarray, patch_size: tuple[int, int]) -> np.ndarray:
    patch_h, patch_w = patch_size
    integral = np.pad(mask, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    return (
        integral[patch_h:, patch_w:]
        - integral[:-patch_h, patch_w:]
        - integral[patch_h:, :-patch_w]
        + integral[:-patch_h, :-patch_w]
    )


def _build_hard_negative_band(vessel_mask: np.ndarray, band_width: int) -> np.ndarray:
    kernel_size = 2 * band_width + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(vessel_mask.astype(np.uint8, copy=False), kernel, iterations=1) > 0
    return dilated & ~vessel_mask


def pad_image_for_patching(
    image: np.ndarray,
    patch_size: tuple[int, int],
    stride: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride
    _, height, width = image.shape

    if height < patch_h:
        pad_h = patch_h - height
    else:
        pad_h = (stride_h - ((height - patch_h) % stride_h)) % stride_h

    if width < patch_w:
        pad_w = patch_w - width
    else:
        pad_w = (stride_w - ((width - patch_w) % stride_w)) % stride_w

    padded = np.pad(
        image,
        pad_width=((0, 0), (0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=0,
    )
    return padded.astype(np.float32, copy=False), (height, width)


def extract_ordered_patches(
    image: np.ndarray,
    patch_size: tuple[int, int],
    stride: tuple[int, int],
) -> np.ndarray:
    image_tensor = torch.from_numpy(image[np.newaxis, ...].astype(np.float32))
    unfolded = F.unfold(image_tensor, kernel_size=patch_size, stride=stride)
    patch_count = unfolded.shape[-1]
    channel_count = image.shape[0]
    patch_h, patch_w = patch_size

    patches = (
        unfolded.transpose(1, 2)
        .reshape(patch_count, channel_count, patch_h, patch_w)
        .contiguous()
        .cpu()
        .numpy()
    )
    return patches


def reconstruct_from_ordered_patches(
    patches: np.ndarray,
    output_shape: tuple[int, int],
    patch_size: tuple[int, int],
    stride: tuple[int, int],
) -> np.ndarray:
    patch_tensor = torch.from_numpy(patches.astype(np.float32))
    patch_count, channels, patch_h, patch_w = patch_tensor.shape

    fold_input = (
        patch_tensor.reshape(1, patch_count, channels, patch_h, patch_w)
        .permute(0, 2, 3, 4, 1)
        .reshape(1, channels * patch_h * patch_w, patch_count)
        .contiguous()
    )
    reconstructed = F.fold(
        fold_input,
        output_size=output_shape,
        kernel_size=patch_size,
        stride=stride,
    )
    normalizer = F.fold(
        torch.ones_like(fold_input),
        output_size=output_shape,
        kernel_size=patch_size,
        stride=stride,
    )
    return (reconstructed / normalizer).squeeze(0).cpu().numpy()
