from __future__ import annotations

import cv2
import numpy as np

from .models import FrameLevelResult, FrameRegistration


ECC_MOTION_MODEL = cv2.MOTION_AFFINE
ECC_METHOD_NAME = "ecc_affine"
ECC_MIN_SKELETON_POINTS = 32
ECC_MAX_ITERATIONS = 100
ECC_TERMINATION_EPS = 1e-5
ECC_BLUR_KERNEL_SIZE = 9


def build_frame_registrations(
    frame_results: list[FrameLevelResult],
    reference_frame: FrameLevelResult,
) -> list[FrameRegistration]:
    registrations: list[FrameRegistration] = []

    for frame_result in frame_results:
        registrations.append(register_frame_to_reference(frame_result, reference_frame))

    return registrations


def register_frame_to_reference(
    frame_result: FrameLevelResult,
    reference_frame: FrameLevelResult,
) -> FrameRegistration:
    if frame_result.image_name == reference_frame.image_name:
        return _build_registration(
            frame_result=frame_result,
            reference_frame=reference_frame,
            transform_matrix=_identity_transform_matrix(),
            method="reference_identity",
            status="success",
            score=1.0,
        )

    fallback_registration = _build_centroid_translation_fallback(
        frame_result,
        reference_frame,
        failure_reason=None,
    )

    if len(frame_result.skeleton_points_xy) < ECC_MIN_SKELETON_POINTS:
        fallback_registration.failure_reason = (
            f"Source frame has insufficient skeleton support for ECC "
            f"({len(frame_result.skeleton_points_xy)} points; requires at least {ECC_MIN_SKELETON_POINTS})."
        )
        return fallback_registration

    if len(reference_frame.skeleton_points_xy) < ECC_MIN_SKELETON_POINTS:
        fallback_registration.failure_reason = (
            f"Reference frame has insufficient skeleton support for ECC "
            f"({len(reference_frame.skeleton_points_xy)} points; requires at least {ECC_MIN_SKELETON_POINTS})."
        )
        return fallback_registration

    reference_support = _render_skeleton_support_image(reference_frame)
    moving_support = _render_skeleton_support_image(frame_result)
    initial_transform = fallback_registration.transform_matrix.copy()

    try:
        score, transform_matrix = cv2.findTransformECC(
            reference_support,
            moving_support,
            initial_transform,
            ECC_MOTION_MODEL,
            _ecc_termination_criteria(),
            None,
        )
    except cv2.error as exc:
        fallback_registration.failure_reason = f"ECC registration failed: {exc}"
        return fallback_registration

    if not _is_valid_transform_matrix(transform_matrix):
        fallback_registration.failure_reason = "ECC registration returned an invalid affine transform."
        return fallback_registration

    return _build_registration(
        frame_result=frame_result,
        reference_frame=reference_frame,
        transform_matrix=transform_matrix,
        method=ECC_METHOD_NAME,
        status="success",
        score=float(score),
    )


def transform_point_to_reference(point_xy: np.ndarray, registration: FrameRegistration) -> np.ndarray:
    transformed_points_xy = transform_points_to_reference(np.asarray([point_xy], dtype=np.float64), registration)
    return transformed_points_xy[0]


def transform_points_to_reference(points_xy: np.ndarray, registration: FrameRegistration) -> np.ndarray:
    if points_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    zero_based_points_xy = _to_zero_based_xy(points_xy.astype(np.float64))
    homogeneous_points = np.column_stack(
        (
            zero_based_points_xy,
            np.ones((len(zero_based_points_xy), 1), dtype=np.float64),
        )
    )
    transformed_zero_based_xy = homogeneous_points @ registration.transform_matrix.T.astype(np.float64)
    return _to_one_based_xy(transformed_zero_based_xy)


def _build_centroid_translation_fallback(
    frame_result: FrameLevelResult,
    reference_frame: FrameLevelResult,
    *,
    failure_reason: str | None,
) -> FrameRegistration:
    reference_centroid = _compute_centroid(reference_frame.skeleton_points_xy)
    frame_centroid = _compute_centroid(frame_result.skeleton_points_xy)

    if reference_centroid is None or frame_centroid is None:
        transform_matrix = _identity_transform_matrix()
        method = "identity_fallback"
        resolved_failure_reason = failure_reason or "Centroid fallback could not be computed because skeleton support is empty."
    else:
        translation_xy = reference_centroid - frame_centroid
        transform_matrix = _translation_transform_matrix(translation_xy)
        method = "centroid_translation_fallback"
        resolved_failure_reason = failure_reason

    return _build_registration(
        frame_result=frame_result,
        reference_frame=reference_frame,
        transform_matrix=transform_matrix,
        method=method,
        status="fallback",
        score=None,
        fallback_used=True,
        failure_reason=resolved_failure_reason,
    )


def _build_registration(
    *,
    frame_result: FrameLevelResult,
    reference_frame: FrameLevelResult,
    transform_matrix: np.ndarray,
    method: str,
    status: str,
    score: float | None,
    fallback_used: bool = False,
    failure_reason: str | None = None,
) -> FrameRegistration:
    return FrameRegistration(
        image_name=frame_result.image_name,
        frame_index=frame_result.frame_index,
        reference_image_name=reference_frame.image_name,
        reference_frame_index=reference_frame.frame_index,
        transform_matrix=transform_matrix.astype(np.float64),
        method=method,
        status=status,
        score=score,
        fallback_used=fallback_used,
        failure_reason=failure_reason,
    )


def _render_skeleton_support_image(frame_result: FrameLevelResult) -> np.ndarray:
    support_image = np.zeros((int(frame_result.height), int(frame_result.width)), dtype=np.float32)
    zero_based_points_xy = _to_zero_based_xy(frame_result.skeleton_points_xy.astype(np.int32))
    if zero_based_points_xy.size == 0:
        return support_image

    valid_points = _filter_valid_points(zero_based_points_xy, width=int(frame_result.width), height=int(frame_result.height))
    if valid_points.size == 0:
        return support_image

    valid_points_int = np.rint(valid_points).astype(np.int32)
    support_image[valid_points_int[:, 1], valid_points_int[:, 0]] = 1.0
    return cv2.GaussianBlur(support_image, (ECC_BLUR_KERNEL_SIZE, ECC_BLUR_KERNEL_SIZE), 0)


def _filter_valid_points(points_xy: np.ndarray, *, width: int, height: int) -> np.ndarray:
    valid_mask = (
        (points_xy[:, 0] >= 0)
        & (points_xy[:, 0] < width)
        & (points_xy[:, 1] >= 0)
        & (points_xy[:, 1] < height)
    )
    return points_xy[valid_mask]


def _compute_centroid(points_xy: np.ndarray) -> np.ndarray | None:
    if points_xy.size == 0:
        return None

    return np.mean(points_xy.astype(np.float64), axis=0)


def _ecc_termination_criteria() -> tuple[int, int, float]:
    return (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        ECC_MAX_ITERATIONS,
        ECC_TERMINATION_EPS,
    )


def _identity_transform_matrix() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


def _translation_transform_matrix(translation_xy: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, float(translation_xy[0])],
            [0.0, 1.0, float(translation_xy[1])],
        ],
        dtype=np.float32,
    )


def _is_valid_transform_matrix(transform_matrix: np.ndarray) -> bool:
    if transform_matrix.shape != (2, 3):
        return False
    if not np.all(np.isfinite(transform_matrix)):
        return False

    linear_component = transform_matrix[:, :2]
    determinant = float(np.linalg.det(linear_component))
    return not np.isclose(determinant, 0.0)


def _to_zero_based_xy(points_xy: np.ndarray) -> np.ndarray:
    return points_xy.astype(np.float64) - 1.0


def _to_one_based_xy(points_xy: np.ndarray) -> np.ndarray:
    return points_xy.astype(np.float64) + 1.0
