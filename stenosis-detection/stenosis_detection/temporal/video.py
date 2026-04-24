from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import FrameLevelResult, LesionObservation, LesionTrack, ViewLevelResult
from .rendering import load_resized_image, to_zero_based_xy


DEFAULT_VIDEO_FPS = 3.0
DEFAULT_VIDEO_FORMAT = "mp4"
VIDEO_FORMATS = ("mp4", "gif")
VIDEO_FILE_SUFFIX = "_demo"
FINAL_LESION_COLOR = (0, 96, 255)
PERSISTENT_LESION_COLOR = (120, 215, 235)
TEXT_BOX_COLOR = (12, 18, 24)
TEXT_COLOR = (245, 248, 252)
MUTED_TEXT_COLOR = (208, 216, 224)
BACKGROUND_COLOR = (24, 28, 32)


def build_view_video_path(output_json_path: str | Path, *, video_format: str = DEFAULT_VIDEO_FORMAT) -> Path:
    resolved_output_path = Path(output_json_path)
    _validate_video_format(video_format)
    return resolved_output_path.with_name(f"{resolved_output_path.stem}{VIDEO_FILE_SUFFIX}.{video_format}")


def save_view_demo_video(
    view_result: ViewLevelResult,
    output_json_path: str | Path,
    *,
    fps: float = DEFAULT_VIDEO_FPS,
    video_format: str = DEFAULT_VIDEO_FORMAT,
) -> Path:
    _validate_video_settings(fps=fps, video_format=video_format)

    frames = create_view_demo_frames(view_result)
    if not frames:
        raise ValueError("Cannot write a temporal demo video without any frames.")

    output_path = build_view_video_path(output_json_path, video_format=video_format)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if video_format == "mp4":
        _write_mp4_video(frames, output_path, fps=fps)
    else:
        _write_gif_video(frames, output_path, fps=fps)

    return output_path


def create_view_demo_frames(view_result: ViewLevelResult) -> list[np.ndarray]:
    tracks_by_id = {track.track_id: track for track in view_result.tracks}
    final_track = None if view_result.final_lesion is None else tracks_by_id.get(view_result.final_lesion.track_id)
    supporting_tracks = [track for track in view_result.tracks if view_result.final_lesion is not None and track.track_id != view_result.final_lesion.track_id]
    persistent_track_ids = {lesion.track_id for lesion in view_result.persistent_lesions}
    subtle_tracks = [track for track in supporting_tracks if track.track_id in persistent_track_ids]

    demo_frames: list[np.ndarray] = []
    total_frame_count = len(view_result.frames)

    for frame_position, frame_result in enumerate(view_result.frames, start=1):
        frame_canvas = _load_video_canvas(frame_result)

        for track in subtle_tracks:
            observation = _find_observation_for_frame(track, frame_result)
            if observation is None:
                continue
            _draw_subtle_lesion_marker(frame_canvas, observation.point_xy)

        final_observation = None if final_track is None else _find_observation_for_frame(final_track, frame_result)
        if final_observation is not None:
            _draw_final_lesion_marker(frame_canvas, final_observation.point_xy)

        _draw_video_overlay(
            frame_canvas,
            view_result,
            frame_result,
            frame_position=frame_position,
            total_frame_count=total_frame_count,
            final_observation=final_observation,
        )
        demo_frames.append(frame_canvas)

    return demo_frames


def _load_video_canvas(frame_result: FrameLevelResult) -> np.ndarray:
    width = int(frame_result.width or 600)
    height = int(frame_result.height or 800)
    canvas = load_resized_image(frame_result.image_path, frame_result, cv2.IMREAD_COLOR)
    if canvas is not None:
        return canvas

    mask_image = load_resized_image(frame_result.mask_path, frame_result, cv2.IMREAD_GRAYSCALE)
    if mask_image is not None:
        return cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

    return np.full((height, width, 3), BACKGROUND_COLOR, dtype=np.uint8)


def _draw_subtle_lesion_marker(canvas: np.ndarray, point_xy: np.ndarray) -> None:
    center_xy = _to_canvas_point(point_xy)
    if center_xy is None:
        return

    cv2.circle(canvas, center_xy, radius=8, color=PERSISTENT_LESION_COLOR, thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, center_xy, radius=2, color=PERSISTENT_LESION_COLOR, thickness=-1, lineType=cv2.LINE_AA)


def _draw_final_lesion_marker(canvas: np.ndarray, point_xy: np.ndarray) -> None:
    center_xy = _to_canvas_point(point_xy)
    if center_xy is None:
        return

    cv2.circle(canvas, center_xy, radius=18, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(canvas, center_xy, radius=11, color=FINAL_LESION_COLOR, thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(canvas, center_xy, radius=3, color=FINAL_LESION_COLOR, thickness=-1, lineType=cv2.LINE_AA)
    cv2.line(canvas, (center_xy[0] - 10, center_xy[1]), (center_xy[0] + 10, center_xy[1]), FINAL_LESION_COLOR, 1, cv2.LINE_AA)
    cv2.line(canvas, (center_xy[0], center_xy[1] - 10), (center_xy[0], center_xy[1] + 10), FINAL_LESION_COLOR, 1, cv2.LINE_AA)


def _draw_video_overlay(
    canvas: np.ndarray,
    view_result: ViewLevelResult,
    frame_result: FrameLevelResult,
    *,
    frame_position: int,
    total_frame_count: int,
    final_observation: LesionObservation | None,
) -> None:
    overlay = canvas.copy()
    cv2.rectangle(overlay, (18, 18), (330, 164), color=TEXT_BOX_COLOR, thickness=-1)
    cv2.addWeighted(overlay, 0.56, canvas, 0.44, 0.0, dst=canvas)

    final_lesion = view_result.final_lesion
    if final_lesion is None:
        lines = [
            _frame_label(frame_result, frame_position=frame_position, total_frame_count=total_frame_count),
            "Fused degree: --",
            "Frame degree: --",
            "Support: 0/0",
            "Severity: none",
        ]
    else:
        frame_degree_text = "--" if final_observation is None else f"{final_observation.degree:.3f}"
        lines = [
            _frame_label(frame_result, frame_position=frame_position, total_frame_count=total_frame_count),
            f"Fused degree: {final_lesion.median_degree:.3f}",
            f"Frame degree: {frame_degree_text}",
            f"Support: {final_lesion.supporting_frame_count}/{final_lesion.total_frame_count}",
            f"Severity: {final_lesion.severity.title()}",
        ]

    y = 46
    for line_index, line in enumerate(lines):
        color = TEXT_COLOR if line_index == 0 else MUTED_TEXT_COLOR
        thickness = 2 if line_index == 0 else 1
        font_scale = 0.64 if line_index == 0 else 0.56
        cv2.putText(canvas, line, (34, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        y += 26 if line_index == 0 else 24

    if frame_result.image_name == view_result.reference_frame.image_name:
        cv2.putText(canvas, "REFERENCE", (34, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.46, FINAL_LESION_COLOR, 1, cv2.LINE_AA)


def _frame_label(frame_result: FrameLevelResult, *, frame_position: int, total_frame_count: int) -> str:
    if frame_result.frame_index is None:
        return f"Frame {frame_position}/{total_frame_count}"
    return f"Frame {frame_result.frame_index} ({frame_position}/{total_frame_count})"


def _find_observation_for_frame(track: LesionTrack, frame_result: FrameLevelResult) -> LesionObservation | None:
    for observation in track.ordered_observations:
        if observation.image_name == frame_result.image_name:
            return observation
        if observation.frame_index is not None and frame_result.frame_index is not None and observation.frame_index == frame_result.frame_index:
            return observation
    return None


def _to_canvas_point(point_xy: np.ndarray) -> tuple[int, int] | None:
    zero_based_xy = to_zero_based_xy(np.asarray([point_xy], dtype=np.float64))
    if zero_based_xy.size == 0:
        return None

    rounded_xy = np.rint(zero_based_xy[0]).astype(np.int32)
    return int(rounded_xy[0]), int(rounded_xy[1])


def _write_mp4_video(frames: list[np.ndarray], output_path: Path, *, fps: float) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Could not open video writer for: {output_path}")

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _write_gif_video(frames: list[np.ndarray], output_path: Path, *, fps: float) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover - depends on optional dependency.
        raise ValueError("GIF output requires imageio. Install imageio or use --video-format mp4.") from exc

    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    imageio.mimsave(output_path, rgb_frames, duration=1.0 / fps, loop=0)


def _validate_video_settings(*, fps: float, video_format: str) -> None:
    if fps <= 0.0:
        raise ValueError("video fps must be positive.")
    _validate_video_format(video_format)


def _validate_video_format(video_format: str) -> None:
    if video_format not in VIDEO_FORMATS:
        raise ValueError(f"Unsupported video format '{video_format}'. Expected one of: {', '.join(VIDEO_FORMATS)}")
