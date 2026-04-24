from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from angio_keyframes.images import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    enhance_vessel_image,
    get_blackhat_kernels,
    get_preprocessing_resources,
    list_image_files,
    load_grayscale_image,
)
from angio_keyframes.models import KeyframeCandidate

BackendName = Literal["cpu", "cuda"]
BACKEND_CHOICES: tuple[BackendName, BackendName] = ("cpu", "cuda")
CUDA_REQUIRED_FUNCTIONS = (
    "createCLAHE",
    "createMorphologyFilter",
    "normalize",
    "subtract",
    "max",
    "maxWithScalar",
    "meanStdDev",
)


def build_baseline_image(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("Cannot build a baseline image from an empty frame list.")
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.float32)


def compute_contrast_fill_score(baseline: np.ndarray, scoring_frame: np.ndarray) -> float:
    delta = scoring_frame.astype(np.float32) - baseline
    return float(np.clip(delta, a_min=0.0, a_max=None).mean())


def get_cuda_unavailable_reason() -> str | None:
    if not hasattr(cv2, "cuda"):
        return "CUDA backend requested, but this OpenCV build does not expose cv2.cuda."

    cuda_module = cv2.cuda
    if not hasattr(cuda_module, "getCudaEnabledDeviceCount"):
        return "CUDA backend requested, but this OpenCV build does not expose CUDA device queries."

    missing_functions = [
        name for name in CUDA_REQUIRED_FUNCTIONS if not hasattr(cuda_module, name)
    ]
    if missing_functions:
        missing = ", ".join(missing_functions)
        return (
            "CUDA backend requested, but the installed OpenCV Python bindings do not expose "
            f"required CUDA functions: {missing}. Rebuild OpenCV with CUDA support."
        )

    try:
        device_count = cuda_module.getCudaEnabledDeviceCount()
    except cv2.error as exc:
        return (
            "CUDA backend requested, but OpenCV reports CUDA support is unavailable: "
            f"{exc}"
        )

    if device_count <= 0:
        return (
            "CUDA backend requested, but no CUDA-capable OpenCV device is available. "
            "Ensure OpenCV is built with CUDA support and a CUDA device is visible."
        )

    return None


def is_cuda_available() -> bool:
    return get_cuda_unavailable_reason() is None


class ScoringBackend(ABC):
    name: BackendName

    def score_frame_directory(
        self,
        frames_dir: Path,
        baseline_frames: int,
    ) -> list[KeyframeCandidate]:
        if baseline_frames <= 0:
            raise ValueError("baseline_frames must be greater than 0.")

        image_paths = list_image_files(frames_dir)
        if not image_paths:
            return []

        scores = self.score_image_paths(image_paths, baseline_frames)
        return [
            KeyframeCandidate(
                frame_index=frame_index,
                name=image_path.name,
                source_path=image_path,
                score=score,
            )
            for frame_index, (image_path, score) in enumerate(zip(image_paths, scores, strict=True))
        ]

    @abstractmethod
    def score_image_paths(self, image_paths: list[Path], baseline_frames: int) -> list[float]:
        raise NotImplementedError


class CpuBackend(ScoringBackend):
    name: BackendName = "cpu"

    def __init__(self) -> None:
        self._resources = get_preprocessing_resources()

    def score_image_paths(self, image_paths: list[Path], baseline_frames: int) -> list[float]:
        baseline_count = min(baseline_frames, len(image_paths))
        baseline_frames_cpu = [
            enhance_vessel_image(load_grayscale_image(image_path), self._resources)
            for image_path in image_paths[:baseline_count]
        ]
        baseline = build_baseline_image(baseline_frames_cpu)

        scores = [
            compute_contrast_fill_score(baseline, scoring_frame)
            for scoring_frame in baseline_frames_cpu
        ]
        for image_path in image_paths[baseline_count:]:
            scoring_frame = enhance_vessel_image(load_grayscale_image(image_path), self._resources)
            scores.append(compute_contrast_fill_score(baseline, scoring_frame))

        return scores


class CudaBackend(ScoringBackend):
    name: BackendName = "cuda"

    def __init__(self) -> None:
        reason = get_cuda_unavailable_reason()
        if reason is not None:
            raise RuntimeError(reason)

        self._cuda = cv2.cuda
        self._clahe = self._cuda.createCLAHE(CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE)
        self._blackhat_filters = tuple(
            self._cuda.createMorphologyFilter(cv2.MORPH_BLACKHAT, cv2.CV_8UC1, kernel)
            for kernel in get_blackhat_kernels()
        )

    def score_image_paths(self, image_paths: list[Path], baseline_frames: int) -> list[float]:
        baseline_count = min(baseline_frames, len(image_paths))
        baseline_frames_cpu: list[np.ndarray] = []
        baseline_frames_gpu: list[Any] = []

        for image_path in image_paths[:baseline_count]:
            enhanced_frame = self._upload_and_enhance(image_path)
            baseline_frames_gpu.append(enhanced_frame)
            baseline_frames_cpu.append(enhanced_frame.download())

        baseline = build_baseline_image(baseline_frames_cpu)
        baseline_gpu = cv2.cuda_GpuMat()
        baseline_gpu.upload(baseline)

        scores = [self._compute_gpu_score(baseline_gpu, frame) for frame in baseline_frames_gpu]
        for image_path in image_paths[baseline_count:]:
            scores.append(self._compute_gpu_score(baseline_gpu, self._upload_and_enhance(image_path)))

        return scores

    def _upload_and_enhance(self, image_path: Path) -> Any:
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(load_grayscale_image(image_path))

        clahe_image = self._apply_gpu_filter(self._clahe, gpu_image)
        responses = [
            self._apply_gpu_filter(filter_, clahe_image)
            for filter_ in self._blackhat_filters
        ]
        combined = responses[0]
        for response in responses[1:]:
            combined = self._cuda_binary_op(self._cuda.max, combined, response)

        return self._cuda_normalize(combined)

    def _compute_gpu_score(self, baseline_gpu: Any, scoring_frame_gpu: Any) -> float:
        scoring_frame_float = scoring_frame_gpu.convertTo(cv2.CV_32F)
        delta = self._cuda_binary_op(
            self._cuda.subtract,
            scoring_frame_float,
            baseline_gpu,
            cv2.noArray(),
            cv2.CV_32F,
        )
        clamped = self._cuda_scalar_op(self._cuda.maxWithScalar, delta, (0.0, 0.0, 0.0, 0.0))

        try:
            mean, _stddev = self._cuda.meanStdDev(clamped)
            return float(mean[0])
        except TypeError:
            stats_gpu = cv2.cuda_GpuMat()
            self._cuda.meanStdDev(clamped, stats_gpu)
            stats_cpu = stats_gpu.download()
            return float(stats_cpu[0, 0])

    @staticmethod
    def _apply_gpu_filter(operator: Any, src: Any) -> Any:
        try:
            return operator.apply(src)
        except TypeError:
            dst = cv2.cuda_GpuMat()
            operator.apply(src, dst)
            return dst

    @staticmethod
    def _cuda_binary_op(operator: Any, src1: Any, src2: Any, *extra_args: Any) -> Any:
        dst = cv2.cuda_GpuMat()
        operator(src1, src2, dst, *extra_args)
        return dst

    @staticmethod
    def _cuda_scalar_op(operator: Any, src: Any, scalar: tuple[float, float, float, float]) -> Any:
        dst = cv2.cuda_GpuMat()
        operator(src, scalar, dst)
        return dst

    def _cuda_normalize(self, src: Any) -> Any:
        dst = cv2.cuda_GpuMat()
        self._cuda.normalize(src, dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return dst


def create_backend(name: BackendName) -> ScoringBackend:
    if name == "cpu":
        return CpuBackend()
    if name == "cuda":
        return CudaBackend()
    raise ValueError(f"Unsupported backend: {name}")
