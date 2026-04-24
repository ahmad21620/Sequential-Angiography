from __future__ import annotations

import numpy as np

try:
    from skimage.morphology import thin as skimage_thin
except ImportError:  # pragma: no cover - exercised only when skimage is absent.
    skimage_thin = None


def thin_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
    binary_mask = np.asarray(binary_mask, dtype=bool)

    if skimage_thin is not None:
        return np.asarray(skimage_thin(binary_mask), dtype=bool)

    return _zhang_suen_thinning(binary_mask)


def _zhang_suen_thinning(binary_mask: np.ndarray) -> np.ndarray:
    image = binary_mask.copy()

    while True:
        first_pass = _zhang_suen_pass(image, first_subiteration=True)
        image[first_pass] = False

        second_pass = _zhang_suen_pass(image, first_subiteration=False)
        image[second_pass] = False

        if not np.any(first_pass) and not np.any(second_pass):
            break

    return image


def _zhang_suen_pass(binary_mask: np.ndarray, *, first_subiteration: bool) -> np.ndarray:
    padded = np.pad(binary_mask, pad_width=1, mode="constant", constant_values=False)

    p2 = padded[:-2, 1:-1]
    p3 = padded[:-2, 2:]
    p4 = padded[1:-1, 2:]
    p5 = padded[2:, 2:]
    p6 = padded[2:, 1:-1]
    p7 = padded[2:, :-2]
    p8 = padded[1:-1, :-2]
    p9 = padded[:-2, :-2]
    center = padded[1:-1, 1:-1]

    neighbor_count = (
        p2.astype(np.uint8)
        + p3.astype(np.uint8)
        + p4.astype(np.uint8)
        + p5.astype(np.uint8)
        + p6.astype(np.uint8)
        + p7.astype(np.uint8)
        + p8.astype(np.uint8)
        + p9.astype(np.uint8)
    )

    transitions = (
        (~p2 & p3).astype(np.uint8)
        + (~p3 & p4).astype(np.uint8)
        + (~p4 & p5).astype(np.uint8)
        + (~p5 & p6).astype(np.uint8)
        + (~p6 & p7).astype(np.uint8)
        + (~p7 & p8).astype(np.uint8)
        + (~p8 & p9).astype(np.uint8)
        + (~p9 & p2).astype(np.uint8)
    )

    common = center & (neighbor_count >= 2) & (neighbor_count <= 6) & (transitions == 1)

    if first_subiteration:
        condition_a = ~(p2 & p4 & p6)
        condition_b = ~(p4 & p6 & p8)
    else:
        condition_a = ~(p2 & p4 & p8)
        condition_b = ~(p2 & p6 & p8)

    return common & condition_a & condition_b
