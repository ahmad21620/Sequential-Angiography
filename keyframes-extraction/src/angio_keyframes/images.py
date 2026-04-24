from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
IGNORED_IMAGE_STEMS = frozenset({".extract_complete"})
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
BLACKHAT_KERNEL_SIZES = (9, 15, 21)


@dataclass(frozen=True)
class PreprocessingResources:
    clahe: cv2.CLAHE
    blackhat_kernels: tuple[np.ndarray, ...]


def is_supported_image_name(name: str) -> bool:
    image_path = Path(name)
    return (
        image_path.suffix.lower() in IMAGE_SUFFIXES
        and image_path.stem.lower() not in IGNORED_IMAGE_STEMS
    )


def is_image_file(path: Path) -> bool:
    return path.is_file() and is_supported_image_name(path.name)


def list_image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if is_image_file(path))


@lru_cache(maxsize=1)
def get_blackhat_kernels() -> tuple[np.ndarray, ...]:
    return tuple(
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        for kernel_size in BLACKHAT_KERNEL_SIZES
    )


@lru_cache(maxsize=1)
def get_preprocessing_resources() -> PreprocessingResources:
    return PreprocessingResources(
        clahe=cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID_SIZE,
        ),
        blackhat_kernels=get_blackhat_kernels(),
    )


def load_grayscale_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def enhance_vessel_image(
    image: np.ndarray,
    resources: PreprocessingResources | None = None,
) -> np.ndarray:
    preprocessing = get_preprocessing_resources() if resources is None else resources
    clahe_image = preprocessing.clahe.apply(image)

    blackhat_responses = [
        cv2.morphologyEx(
            clahe_image,
            cv2.MORPH_BLACKHAT,
            kernel,
        )
        for kernel in preprocessing.blackhat_kernels
    ]
    combined_response = np.maximum.reduce(blackhat_responses)

    return cv2.normalize(
        combined_response,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )


def write_grayscale_image(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not write image: {output_path}")
