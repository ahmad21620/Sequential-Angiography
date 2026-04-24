from __future__ import annotations

import cv2
import numpy as np


def rgb_to_grayscale(images: np.ndarray) -> np.ndarray:
    if images.shape[1] == 1:
        return images.astype(np.float32, copy=True)
    if images.shape[1] != 3:
        raise ValueError(
            f"Expected images with 1 or 3 channels, got shape {images.shape}."
        )

    gray = (
        images[:, 0, ...] * 0.299
        + images[:, 1, ...] * 0.587
        + images[:, 2, ...] * 0.114
    )
    return gray[:, np.newaxis, ...]


def normalize_images(images: np.ndarray) -> np.ndarray:
    images = images.astype(np.float32, copy=True)
    mean = float(images.mean())
    std = float(images.std())
    if std == 0.0:
        std = 1.0

    normalized = (images - mean) / std
    for index in range(normalized.shape[0]):
        image_min = float(normalized[index].min())
        image_max = float(normalized[index].max())
        if image_max == image_min:
            normalized[index] = 0.0
        else:
            normalized[index] = (normalized[index] - image_min) / (image_max - image_min)

    return normalized * 255.0


def clahe_equalize(images: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    equalized = images.astype(np.float32, copy=True)
    for index in range(equalized.shape[0]):
        equalized[index, 0] = clahe.apply(equalized[index, 0].astype(np.uint8))
    return equalized


def adjust_gamma(images: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    lookup_table = np.array(
        [((value / 255.0) ** inv_gamma) * 255.0 for value in range(256)],
        dtype=np.uint8,
    )

    adjusted = images.astype(np.float32, copy=True)
    for index in range(adjusted.shape[0]):
        adjusted[index, 0] = cv2.LUT(adjusted[index, 0].astype(np.uint8), lookup_table)

    return adjusted


def preprocess_images(images: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    grayscale = rgb_to_grayscale(images)
    normalized = normalize_images(grayscale)
    equalized = clahe_equalize(normalized)
    adjusted = adjust_gamma(equalized, gamma=gamma)
    return adjusted.astype(np.float32) / 255.0


def preprocess_image(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    return preprocess_images(np.expand_dims(image, axis=0), gamma=gamma)[0]


def binarize_masks(masks: np.ndarray) -> np.ndarray:
    return (masks > 0).astype(np.float32)
