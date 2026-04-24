from __future__ import annotations

import cv2
import numpy as np

from .config import AugmentationConfig


class PatchAugmenter:
    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *aligned_tensors: np.ndarray,
        seed: int | None = None,
    ) -> tuple[np.ndarray, ...]:
        if not self.config.enable_augmentation:
            return self._finalize_outputs(image, mask, aligned_tensors)

        rng = np.random.default_rng(seed)
        image_tensor = image.astype(np.float32, copy=False)
        aligned_tensors_list = [mask.astype(np.float32, copy=False)] + [
            tensor.astype(np.float32, copy=False) for tensor in aligned_tensors
        ]

        image_tensor, aligned_tensors_list = self._apply_spatial_transforms(
            image_tensor,
            aligned_tensors_list,
            rng,
        )
        image_tensor = self._apply_photometric_transforms(image_tensor, rng)
        image_tensor = self._apply_image_degradations(image_tensor, rng)
        return self._finalize_outputs(
            image_tensor,
            aligned_tensors_list[0],
            aligned_tensors_list[1:],
        )

    def _apply_spatial_transforms(
        self,
        image: np.ndarray,
        aligned_tensors: list[np.ndarray],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        if rng.random() < self.config.hflip_prob:
            image = image[:, :, ::-1]
            aligned_tensors = [tensor[:, :, ::-1] for tensor in aligned_tensors]

        if rng.random() < self.config.vflip_prob:
            image = image[:, ::-1, :]
            aligned_tensors = [tensor[:, ::-1, :] for tensor in aligned_tensors]

        if rng.random() < self.config.rot90_prob:
            rotation_k = self._sample_rot90_k(image.shape[-2], image.shape[-1], rng)
            image = np.rot90(image, k=rotation_k, axes=(-2, -1))
            aligned_tensors = [
                np.rot90(tensor, k=rotation_k, axes=(-2, -1))
                for tensor in aligned_tensors
            ]

        if self.config.max_rotation_deg > 0.0 and rng.random() < self.config.rotation_prob:
            angle = float(
                rng.uniform(-self.config.max_rotation_deg, self.config.max_rotation_deg)
            )
            image = self._rotate_tensor(image, angle, interpolation=cv2.INTER_LINEAR)
            aligned_tensors = [
                self._rotate_tensor(tensor, angle, interpolation=cv2.INTER_NEAREST)
                for tensor in aligned_tensors
            ]

        return image, aligned_tensors

    def _apply_photometric_transforms(
        self,
        image: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        augmented = image.astype(np.float32, copy=False)

        if self.config.brightness_jitter > 0.0:
            brightness_delta = float(
                rng.uniform(-self.config.brightness_jitter, self.config.brightness_jitter)
            )
            augmented = augmented + brightness_delta

        if self.config.contrast_jitter > 0.0:
            contrast_factor = float(
                rng.uniform(
                    1.0 - self.config.contrast_jitter,
                    1.0 + self.config.contrast_jitter,
                )
            )
            patch_mean = augmented.mean(axis=(-2, -1), keepdims=True)
            augmented = patch_mean + contrast_factor * (augmented - patch_mean)

        if self.config.gamma_jitter > 0.0:
            gamma = float(
                rng.uniform(
                    max(0.1, 1.0 - self.config.gamma_jitter),
                    1.0 + self.config.gamma_jitter,
                )
            )
            augmented = np.power(np.clip(augmented, 0.0, 1.0), gamma).astype(
                np.float32,
                copy=False,
            )

        return augmented

    def _apply_image_degradations(
        self,
        image: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        degraded = image.astype(np.float32, copy=False)

        if self.config.noise_std > 0.0 and rng.random() < self.config.noise_prob:
            noise = rng.normal(
                loc=0.0,
                scale=self.config.noise_std,
                size=degraded.shape,
            ).astype(np.float32)
            degraded = degraded + noise

        if self.config.blur_kernel_size > 1 and rng.random() < self.config.blur_prob:
            degraded = np.stack(
                [
                    cv2.GaussianBlur(
                        channel,
                        (self.config.blur_kernel_size, self.config.blur_kernel_size),
                        sigmaX=0.0,
                    )
                    for channel in degraded
                ],
                axis=0,
            ).astype(np.float32, copy=False)

        return degraded

    def _rotate_tensor(
        self,
        tensor: np.ndarray,
        angle: float,
        interpolation: int,
    ) -> np.ndarray:
        channels, height, width = tensor.shape
        center = ((width - 1) * 0.5, (height - 1) * 0.5)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return np.stack(
            [
                cv2.warpAffine(
                    np.ascontiguousarray(channel),
                    rotation_matrix,
                    (width, height),
                    flags=interpolation,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                for channel in tensor
            ],
            axis=0,
        ).astype(np.float32, copy=False)

    def _sample_rot90_k(
        self,
        height: int,
        width: int,
        rng: np.random.Generator,
    ) -> int:
        if height == width:
            return int(rng.integers(1, 4))
        return 2

    def _finalize_outputs(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        aligned_tensors: tuple[np.ndarray, ...] | list[np.ndarray],
    ) -> tuple[np.ndarray, ...]:
        finalized_image = np.ascontiguousarray(
            np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
        )
        finalized_mask = np.ascontiguousarray((mask >= 0.5).astype(np.float32, copy=False))
        finalized_tensors = tuple(
            np.ascontiguousarray(tensor.astype(np.float32, copy=False))
            for tensor in aligned_tensors
        )
        return (finalized_image, finalized_mask, *finalized_tensors)
