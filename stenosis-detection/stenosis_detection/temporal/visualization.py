from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import FrameLevelResult, LesionTrack, PersistentLesion, ViewLevelResult
from .rendering import load_resized_image, paint_points, to_zero_based_xy


SUMMARY_IMAGE_SUFFIX = "_summary.png"
INFO_PANEL_WIDTH = 360
TEXT_MARGIN = 24
LINE_HEIGHT = 30
TITLE_LINE_HEIGHT = 42
TRACK_COLORS = [
    (60, 190, 255),
    (90, 230, 120),
    (255, 190, 60),
    (180, 120, 255),
]
FINAL_LESION_COLOR = (0, 96, 255)
CENTERLINE_COLOR = (255, 220, 40)
MASK_TINT_COLOR = (60, 170, 255)
BACKGROUND_COLOR = (24, 28, 32)
INFO_PANEL_COLOR = (245, 248, 252)
TEXT_COLOR = (28, 34, 42)
MUTED_TEXT_COLOR = (88, 96, 108)


def build_view_visualization_paths(output_json_path: str | Path) -> dict[str, Path]:
    resolved_output_path = Path(output_json_path)
    return {
        "summary_png": resolved_output_path.with_name(f"{resolved_output_path.stem}{SUMMARY_IMAGE_SUFFIX}"),
    }


def save_view_visualization_outputs(
    view_result: ViewLevelResult,
    output_json_path: str | Path,
) -> dict[str, Path]:
    output_paths = build_view_visualization_paths(output_json_path)
    summary_image = create_view_summary_visualization(view_result)

    summary_path = output_paths["summary_png"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(summary_path), summary_image)
    return output_paths


def create_view_summary_visualization(view_result: ViewLevelResult) -> np.ndarray:
    tracks_by_id = _tracks_by_id(view_result.tracks)
    persistent_tracks = [(lesion, tracks_by_id.get(lesion.track_id)) for lesion in view_result.persistent_lesions]

    summary_canvas = _load_reference_canvas(view_result.reference_frame)
    _draw_reference_header(summary_canvas, view_result.reference_frame)
    _draw_centerline(summary_canvas, view_result.reference_frame.skeleton_points_xy)
    _draw_persistent_tracks(summary_canvas, persistent_tracks, final_track_id=_final_track_id(view_result))
    _draw_final_lesion_highlight(summary_canvas, view_result.final_lesion)

    info_panel = _build_info_panel(view_result, persistent_tracks, height=summary_canvas.shape[0])
    return np.hstack((summary_canvas, info_panel))


def _load_reference_canvas(reference_frame: FrameLevelResult) -> np.ndarray:
    width = int(reference_frame.width or 600)
    height = int(reference_frame.height or 800)
    canvas = load_resized_image(reference_frame.image_path, reference_frame, cv2.IMREAD_COLOR)
    mask_image = load_resized_image(reference_frame.mask_path, reference_frame, cv2.IMREAD_GRAYSCALE)

    if canvas is None and mask_image is not None:
        canvas = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
    if canvas is None:
        canvas = np.full((height, width, 3), BACKGROUND_COLOR, dtype=np.uint8)

    if mask_image is not None:
        canvas = _apply_mask_tint(canvas, mask_image)

    return canvas


def _draw_reference_header(canvas: np.ndarray, reference_frame: FrameLevelResult) -> None:
    overlay = canvas.copy()
    cv2.rectangle(overlay, (14, 14), (330, 96), color=(12, 18, 24), thickness=-1)
    cv2.addWeighted(overlay, 0.58, canvas, 0.42, 0.0, dst=canvas)
    cv2.putText(
        canvas,
        "Reference Frame",
        (28, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        reference_frame.image_name,
        (28, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (220, 230, 238),
        1,
        cv2.LINE_AA,
    )


def _draw_centerline(canvas: np.ndarray, skeleton_points_xy: np.ndarray) -> None:
    centerline_overlay = canvas.copy()
    paint_points(centerline_overlay, skeleton_points_xy, color=CENTERLINE_COLOR)
    cv2.addWeighted(centerline_overlay, 0.55, canvas, 0.45, 0.0, dst=canvas)


def _draw_persistent_tracks(
    canvas: np.ndarray,
    persistent_tracks: list[tuple[PersistentLesion, LesionTrack | None]],
    *,
    final_track_id: int | None,
) -> None:
    for index, (lesion, track) in enumerate(persistent_tracks):
        if track is None:
            continue

        color = FINAL_LESION_COLOR if lesion.track_id == final_track_id else TRACK_COLORS[index % len(TRACK_COLORS)]
        _draw_track(canvas, track, color=color, emphasize=lesion.track_id == final_track_id)
        _draw_lesion_label(canvas, lesion, color=color)


def _draw_track(canvas: np.ndarray, track: LesionTrack, *, color: tuple[int, int, int], emphasize: bool) -> None:
    track_points_xy = to_zero_based_xy(track.registered_points_xy)
    if track_points_xy.shape[0] >= 2:
        polyline = np.rint(track_points_xy).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            canvas,
            [polyline],
            isClosed=False,
            color=color,
            thickness=3 if emphasize else 2,
            lineType=cv2.LINE_AA,
        )

    for point_xy in track_points_xy:
        rounded_point_xy = tuple(np.rint(point_xy).astype(np.int32))
        cv2.circle(canvas, rounded_point_xy, radius=5 if emphasize else 4, color=color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, rounded_point_xy, radius=7 if emphasize else 5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


def _draw_lesion_label(canvas: np.ndarray, lesion: PersistentLesion, *, color: tuple[int, int, int]) -> None:
    anchor_point_xy = _best_lesion_anchor_point(lesion)
    if anchor_point_xy is None:
        return

    zero_based_anchor_xy = to_zero_based_xy(np.asarray([anchor_point_xy], dtype=np.float64))[0]
    x = int(np.rint(zero_based_anchor_xy[0])) + 10
    y = int(np.rint(zero_based_anchor_xy[1])) - 10
    cv2.putText(
        canvas,
        f"L{lesion.lesion_id}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_final_lesion_highlight(canvas: np.ndarray, final_lesion: PersistentLesion | None) -> None:
    anchor_point_xy = _best_lesion_anchor_point(final_lesion)
    if anchor_point_xy is None:
        return

    zero_based_anchor_xy = to_zero_based_xy(np.asarray([anchor_point_xy], dtype=np.float64))[0]
    center = tuple(np.rint(zero_based_anchor_xy).astype(np.int32))
    cv2.circle(canvas, center, radius=16, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(canvas, center, radius=10, color=FINAL_LESION_COLOR, thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(canvas, center, radius=3, color=FINAL_LESION_COLOR, thickness=-1, lineType=cv2.LINE_AA)


def _build_info_panel(
    view_result: ViewLevelResult,
    persistent_tracks: list[tuple[PersistentLesion, LesionTrack | None]],
    *,
    height: int,
) -> np.ndarray:
    panel = np.full((height, INFO_PANEL_WIDTH, 3), INFO_PANEL_COLOR, dtype=np.uint8)
    y = TEXT_MARGIN + 6

    y = _draw_text_block(panel, "Temporal Fusion", y, font_scale=0.98, color=TEXT_COLOR, thickness=2, line_height=TITLE_LINE_HEIGHT)
    y = _draw_text_block(panel, f"View: {view_result.view_id}", y, font_scale=0.54, color=MUTED_TEXT_COLOR)
    y = _draw_text_block(
        panel,
        f"Reference: {view_result.reference_frame.image_name}",
        y,
        font_scale=0.54,
        color=MUTED_TEXT_COLOR,
        extra_spacing=12,
    )

    final_lesion = view_result.final_lesion
    y = _draw_section_title(panel, "Final Lesion", y)
    if final_lesion is None:
        y = _draw_text_block(panel, "No persistent lesion passed the fusion thresholds.", y, font_scale=0.55, color=TEXT_COLOR, extra_spacing=12)
    else:
        y = _draw_key_value(panel, "Track", f"{final_lesion.track_id}", y)
        y = _draw_key_value(panel, "Support", f"{final_lesion.supporting_frame_count}/{final_lesion.total_frame_count}", y)
        y = _draw_key_value(panel, "Persistence", f"{final_lesion.persistence_ratio:.2%}", y)
        y = _draw_key_value(panel, "Median Fused Degree", f"{final_lesion.median_degree:.3f}", y)
        y = _draw_key_value(panel, "Max Frame Degree", f"{final_lesion.max_degree:.3f}", y)
        y = _draw_key_value(panel, "Severity", final_lesion.severity.title(), y, extra_spacing=10)

    y = _draw_section_title(panel, "Persistent Lesions", y)
    if not view_result.persistent_lesions:
        y = _draw_text_block(panel, "None", y, font_scale=0.55, color=TEXT_COLOR, extra_spacing=10)
    else:
        for index, (lesion, track) in enumerate(persistent_tracks):
            color = FINAL_LESION_COLOR if final_lesion is not None and lesion.track_id == final_lesion.track_id else TRACK_COLORS[index % len(TRACK_COLORS)]
            y = _draw_lesion_summary_line(panel, lesion, track, y, color=color)
            if y > height - 110:
                break

    y = _draw_section_title(panel, "Fusion", y)
    fallback_count = sum(1 for registration in view_result.registrations if registration.fallback_used)
    y = _draw_key_value(panel, "Tracks", f"{len(view_result.tracks)}", y)
    y = _draw_key_value(panel, "Fallback Registrations", f"{fallback_count}", y)
    y = _draw_key_value(panel, "Rule", "median degree -> max degree -> persistence", y, font_scale=0.46)
    return panel


def _draw_section_title(panel: np.ndarray, title: str, y: int) -> int:
    cv2.putText(panel, title, (TEXT_MARGIN, y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.line(panel, (TEXT_MARGIN, y + 10), (INFO_PANEL_WIDTH - TEXT_MARGIN, y + 10), color=(220, 226, 232), thickness=1)
    return y + 34


def _draw_key_value(
    panel: np.ndarray,
    key: str,
    value: str,
    y: int,
    *,
    font_scale: float = 0.54,
    extra_spacing: int = 0,
) -> int:
    cv2.putText(panel, key, (TEXT_MARGIN, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MUTED_TEXT_COLOR, 1, cv2.LINE_AA)
    for line in _wrap_text(value, max_chars=26):
        cv2.putText(panel, line, (TEXT_MARGIN, y + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR, 1, cv2.LINE_AA)
        y += LINE_HEIGHT
    return y + 8 + extra_spacing


def _draw_text_block(
    panel: np.ndarray,
    text: str,
    y: int,
    *,
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    line_height: int = LINE_HEIGHT,
    extra_spacing: int = 0,
) -> int:
    for line in _wrap_text(text, max_chars=34):
        cv2.putText(panel, line, (TEXT_MARGIN, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        y += line_height
    return y + extra_spacing


def _draw_lesion_summary_line(
    panel: np.ndarray,
    lesion: PersistentLesion,
    track: LesionTrack | None,
    y: int,
    *,
    color: tuple[int, int, int],
) -> int:
    cv2.circle(panel, (TEXT_MARGIN + 6, y - 6), radius=6, color=color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.putText(
        panel,
        f"L{lesion.lesion_id} / T{lesion.track_id}",
        (TEXT_MARGIN + 24, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    stats_text = (
        f"{lesion.supporting_frame_count}/{lesion.total_frame_count}  "
        f"med {lesion.median_degree:.3f}  max {lesion.max_degree:.3f}"
    )
    cv2.putText(
        panel,
        stats_text,
        (TEXT_MARGIN + 24, y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        MUTED_TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    if track is not None and track.mean_centerline_position is not None:
        cv2.putText(
            panel,
            f"pos {track.mean_centerline_position:.3f}",
            (TEXT_MARGIN + 24, y + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            MUTED_TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        return y + 58

    return y + 46


def _tracks_by_id(tracks: list[LesionTrack]) -> dict[int, LesionTrack]:
    return {track.track_id: track for track in tracks}


def _final_track_id(view_result: ViewLevelResult) -> int | None:
    if view_result.final_lesion is None:
        return None
    return view_result.final_lesion.track_id


def _best_lesion_anchor_point(lesion: PersistentLesion | None) -> np.ndarray | None:
    if lesion is None:
        return None
    if lesion.median_centerline_point_xy is not None:
        return lesion.median_centerline_point_xy.astype(np.float64)
    if lesion.median_registered_point_xy is not None:
        return lesion.median_registered_point_xy.astype(np.float64)
    return None


def _apply_mask_tint(canvas: np.ndarray, mask_image: np.ndarray) -> np.ndarray:
    colored_mask = np.zeros_like(canvas)
    colored_mask[:, :, 0] = np.clip(mask_image.astype(np.uint16) * MASK_TINT_COLOR[0] // 255, 0, 255).astype(np.uint8)
    colored_mask[:, :, 1] = np.clip(mask_image.astype(np.uint16) * MASK_TINT_COLOR[1] // 255, 0, 255).astype(np.uint8)
    colored_mask[:, :, 2] = np.clip(mask_image.astype(np.uint16) * MASK_TINT_COLOR[2] // 255, 0, 255).astype(np.uint8)
    return cv2.addWeighted(canvas, 0.9, colored_mask, 0.22, 0.0)


def _wrap_text(text: str, *, max_chars: int) -> list[str]:
    words = []
    for raw_word in text.split():
        if len(raw_word) <= max_chars:
            words.append(raw_word)
            continue

        for start_index in range(0, len(raw_word), max_chars):
            words.append(raw_word[start_index : start_index + max_chars])

    if not words:
        return [""]

    lines: list[str] = []
    current_line = words[0]

    for word in words[1:]:
        if len(current_line) + 1 + len(word) <= max_chars:
            current_line = f"{current_line} {word}"
            continue
        lines.append(current_line)
        current_line = word

    lines.append(current_line)
    return lines
