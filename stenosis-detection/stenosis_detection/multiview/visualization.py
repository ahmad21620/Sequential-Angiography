from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .fusion import compute_angle_distance, compute_view_diversity_weight
from .models import MultiViewCaseResult, MultiViewPerViewSummary, MultiViewViewInput


SUMMARY_IMAGE_SUFFIX = "_summary.png"
SUPPORT_MATRIX_IMAGE_SUFFIX = "_support_matrix.png"
TEMPORAL_SUMMARY_IMAGE_SUFFIX = "_summary.png"

CANVAS_WIDTH = 1600
MARGIN = 32
PANEL_GAP = 24
BACKGROUND_COLOR = (24, 28, 33)
PANEL_COLOR = (246, 248, 252)
PANEL_DARK_COLOR = (34, 40, 48)
ROW_ALT_COLOR = (237, 241, 246)
BORDER_COLOR = (214, 222, 232)
TEXT_COLOR = (31, 38, 48)
MUTED_TEXT_COLOR = (92, 103, 118)
WHITE_COLOR = (255, 255, 255)
PRIMARY_COLOR = (255, 132, 48)
DISTINCT_COLOR = (92, 204, 128)
SIMILAR_COLOR = (66, 178, 245)
CANDIDATE_COLOR = (92, 210, 235)
NO_CANDIDATE_COLOR = (150, 158, 168)
GRID_COLOR = (210, 218, 228)


@dataclass(slots=True)
class _ViewEvidence:
    summary: MultiViewPerViewSummary
    role: str
    role_label: str
    candidate_score: float | None
    contribution: float
    angle_distance_to_primary: float | None
    diversity_weight: float | None


def build_multiview_visualization_paths(output_json_path: str | Path) -> dict[str, Path]:
    resolved_output_path = Path(output_json_path)
    return {
        "summary_png": resolved_output_path.with_name(f"{resolved_output_path.stem}{SUMMARY_IMAGE_SUFFIX}"),
        "support_matrix_png": resolved_output_path.with_name(f"{resolved_output_path.stem}{SUPPORT_MATRIX_IMAGE_SUFFIX}"),
    }


def save_multiview_visualization_outputs(
    case_result: MultiViewCaseResult,
    output_json_path: str | Path,
) -> dict[str, Path]:
    """Write case-level multiview summary images next to the fusion JSON."""
    output_paths = build_multiview_visualization_paths(output_json_path)
    for output_path in output_paths.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_image = create_multiview_summary_visualization(case_result)
    support_matrix_image = create_multiview_support_matrix_visualization(case_result)

    if not cv2.imwrite(str(output_paths["summary_png"]), summary_image):
        raise OSError(f"Failed to write multiview summary visualization: {output_paths['summary_png']}")
    if not cv2.imwrite(str(output_paths["support_matrix_png"]), support_matrix_image):
        raise OSError(f"Failed to write multiview support matrix visualization: {output_paths['support_matrix_png']}")

    return output_paths


def create_multiview_summary_visualization(case_result: MultiViewCaseResult) -> np.ndarray:
    """Build a compact visual explanation of the case-level multiview fusion result."""
    evidence_rows = _build_view_evidence(case_result)
    height = max(980, 880 + (len(evidence_rows) * 108))
    canvas = np.full((height, CANVAS_WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)

    _draw_header(canvas, case_result, evidence_rows)
    _draw_angle_map(canvas, case_result, evidence_rows, (MARGIN, 176, 700, 480))
    _draw_conclusion_panel(canvas, case_result, evidence_rows, (MARGIN + 700 + PANEL_GAP, 176, 812, 480))
    _draw_view_table(
        canvas,
        evidence_rows,
        (MARGIN, 684, CANVAS_WIDTH - (2 * MARGIN), height - 716),
        uses_projection_metadata=_uses_projection_view_metadata(case_result),
    )
    return canvas


def create_multiview_support_matrix_visualization(case_result: MultiViewCaseResult) -> np.ndarray:
    """Build a pairwise view-diversity-weight matrix for the available views."""
    views = case_result.views
    view_count = max(1, len(views))
    cell_size = max(44, min(86, 920 // view_count))
    left_margin = 170
    top_margin = 150
    width = max(980, left_margin + (view_count * cell_size) + 90)
    height = top_margin + (view_count * cell_size) + 150
    canvas = np.full((height, width, 3), PANEL_COLOR, dtype=np.uint8)
    uses_projection_metadata = _uses_projection_view_metadata(case_result)

    matrix_title = (
        "Multi-View Projection Support Matrix"
        if uses_projection_metadata
        else "Multi-View Angle Support Matrix"
    )
    matrix_description = (
        "Cells show the projection-group multiplier used when one view supports another. Opposite-side or unknown groups provide no support."
        if uses_projection_metadata
        else "Cells show the angle-diversity multiplier used when one view supports another. Similar views carry less support; distinct views carry stronger support."
    )
    _put_text(canvas, matrix_title, (34, 48), 0.9, TEXT_COLOR, 2)
    _draw_wrapped_text(
        canvas,
        matrix_description,
        34,
        82,
        width - 68,
        0.5,
        MUTED_TEXT_COLOR,
        line_height=24,
    )

    config = case_result.fusion_metadata.config
    for index, view in enumerate(views):
        top_label = _truncate_text(view.view_id, cell_size - 8, 0.42, 1)
        side_label = _truncate_text(view.view_id, left_margin - 64, 0.42, 1)
        x = left_margin + (index * cell_size)
        y = top_margin + (index * cell_size)
        _put_text(canvas, top_label, (x + 4, top_margin - 18), 0.42, MUTED_TEXT_COLOR, 1)
        _put_text(canvas, side_label, (34, y + (cell_size // 2) + 6), 0.42, MUTED_TEXT_COLOR, 1)

    for row_index, row_view in enumerate(views):
        for column_index, column_view in enumerate(views):
            x = left_margin + (column_index * cell_size)
            y = top_margin + (row_index * cell_size)
            if row_index == column_index:
                color = (218, 224, 232)
                label = "-"
            else:
                weight = compute_view_diversity_weight(
                    row_view,
                    column_view,
                    config.view_diversity_mode,
                    duplicate_distance_degrees=config.duplicate_view_angle_distance_degrees,
                    distinct_distance_degrees=config.distinct_view_angle_distance_degrees,
                )
                color = _weight_color(weight)
                label = f"{weight:.2f}" if cell_size >= 54 else f"{weight:.1f}"

            cv2.rectangle(canvas, (x, y), (x + cell_size - 2, y + cell_size - 2), color, thickness=-1)
            cv2.rectangle(canvas, (x, y), (x + cell_size - 2, y + cell_size - 2), BORDER_COLOR, thickness=1)
            text_width = _text_width(label, 0.43, 1)
            _put_text(
                canvas,
                label,
                (x + ((cell_size - text_width) // 2), y + (cell_size // 2) + 7),
                0.43,
                TEXT_COLOR,
                1,
            )

    legend_y = height - 64
    _draw_matrix_legend(
        canvas,
        34,
        legend_y,
        uses_projection_metadata,
        config.duplicate_view_angle_distance_degrees,
        config.distinct_view_angle_distance_degrees,
    )
    return canvas


def _build_view_evidence(case_result: MultiViewCaseResult) -> list[_ViewEvidence]:
    final_lesion = case_result.final_case_lesion
    config = case_result.fusion_metadata.config
    uses_projection_metadata = _uses_projection_view_metadata(case_result)
    raw_support_by_view_id: dict[str, float] = {}
    distance_by_view_id: dict[str, float] = {}
    weight_by_view_id: dict[str, float] = {}

    if final_lesion is not None:
        primary_view = _find_view(case_result.views, final_lesion.primary_view_id)
        if primary_view is not None:
            for summary in case_result.per_view_summary:
                view_id = summary.view_input.view_id
                if view_id == final_lesion.primary_view_id or view_id not in final_lesion.supporting_view_ids:
                    continue
                if summary.best_candidate_score is None:
                    continue

                angle_distance = compute_angle_distance(primary_view, summary.view_input)
                diversity_weight = compute_view_diversity_weight(
                    primary_view,
                    summary.view_input,
                    config.view_diversity_mode,
                    duplicate_distance_degrees=config.duplicate_view_angle_distance_degrees,
                    distinct_distance_degrees=config.distinct_view_angle_distance_degrees,
                )
                raw_support_by_view_id[view_id] = (
                    config.support_score_scale
                    * summary.best_candidate_score.candidate_score
                    * diversity_weight
                )
                distance_by_view_id[view_id] = angle_distance
                weight_by_view_id[view_id] = diversity_weight

    raw_support_total = sum(raw_support_by_view_id.values())
    support_scale = 1.0
    if final_lesion is not None and raw_support_total > 0.0:
        support_scale = min(1.0, final_lesion.distinct_view_support_score / raw_support_total)

    evidence_rows: list[_ViewEvidence] = []
    for summary in case_result.per_view_summary:
        view_id = summary.view_input.view_id
        candidate_score = None if summary.best_candidate_score is None else summary.best_candidate_score.candidate_score
        contribution = 0.0

        if final_lesion is None:
            role = "candidate" if summary.best_candidate is not None else "none"
        elif view_id == final_lesion.primary_view_id:
            role = "primary"
            contribution = final_lesion.score_breakdown.candidate_score
        elif view_id in final_lesion.distinct_supporting_view_ids:
            role = "distinct_support"
            contribution = raw_support_by_view_id.get(view_id, 0.0) * support_scale
        elif view_id in final_lesion.supporting_view_ids:
            role = "similar_support"
            contribution = raw_support_by_view_id.get(view_id, 0.0) * support_scale
        elif summary.best_candidate is not None:
            role = "candidate"
        else:
            role = "none"

        evidence_rows.append(
            _ViewEvidence(
                summary=summary,
                role=role,
                role_label=_role_label(role, uses_projection_metadata),
                candidate_score=candidate_score,
                contribution=contribution,
                angle_distance_to_primary=distance_by_view_id.get(view_id),
                diversity_weight=weight_by_view_id.get(view_id),
            )
        )

    return evidence_rows


def _draw_header(canvas: np.ndarray, case_result: MultiViewCaseResult, evidence_rows: list[_ViewEvidence]) -> None:
    x, y, width, height = MARGIN, 24, CANVAS_WIDTH - (2 * MARGIN), 124
    cv2.rectangle(canvas, (x, y), (x + width, y + height), PANEL_DARK_COLOR, thickness=-1)
    _put_text(canvas, "Multi-View Fusion", (x + 26, y + 42), 1.05, WHITE_COLOR, 2)
    _put_text(canvas, f"Case: {case_result.case_id}", (x + 26, y + 78), 0.58, (218, 228, 238), 1)
    support_text = (
        "CADICA projection groups provide categorical cross-view support."
        if _uses_projection_view_metadata(case_result)
        else "Views with distinct angles provide stronger support; similar-angle views provide weaker support."
    )
    _put_text(
        canvas,
        support_text,
        (x + 26, y + 106),
        0.47,
        (190, 202, 216),
        1,
    )

    final_lesion = case_result.final_case_lesion
    final_label = "No case lesion" if final_lesion is None else final_lesion.severity.title()
    score_text = f"{case_result.confidence.score:.3f}"
    contributing_count = sum(1 for row in evidence_rows if row.contribution > 0.0)

    stat_x = x + width - 620
    _draw_header_stat(canvas, "Final", final_label, stat_x, y + 32, PRIMARY_COLOR)
    _draw_header_stat(canvas, "Confidence", f"{case_result.confidence.label.title()} / {score_text}", stat_x + 190, y + 32, DISTINCT_COLOR)
    _draw_header_stat(canvas, "Views", str(case_result.view_count), stat_x + 410, y + 32, WHITE_COLOR)
    _draw_header_stat(canvas, "Contributing", str(contributing_count), stat_x + 520, y + 32, WHITE_COLOR)


def _draw_header_stat(
    canvas: np.ndarray,
    label: str,
    value: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    _put_text(canvas, label, (x, y), 0.42, (178, 190, 204), 1)
    _put_text(canvas, _truncate_text(value, 150, 0.62, 2), (x, y + 34), 0.62, color, 2)


def _draw_angle_map(
    canvas: np.ndarray,
    case_result: MultiViewCaseResult,
    evidence_rows: list[_ViewEvidence],
    rect: tuple[int, int, int, int],
) -> None:
    x, y, width, height = rect
    _draw_panel(canvas, rect)
    if _uses_projection_view_metadata(case_result):
        _put_text(canvas, "Projection Group Map", (x + 24, y + 38), 0.72, TEXT_COLOR, 2)
        _put_text(canvas, "Numeric angles are placeholders; support follows CADICA projection groups.", (x + 24, y + 66), 0.44, MUTED_TEXT_COLOR, 1)
    else:
        _put_text(canvas, "View-Angle Map", (x + 24, y + 38), 0.72, TEXT_COLOR, 2)
        _put_text(canvas, "x = RAO(-) / LAO(+), y = CAU(-) / CRA(+)", (x + 24, y + 66), 0.44, MUTED_TEXT_COLOR, 1)

    plot_x = x + 78
    plot_y = y + 88
    plot_width = width - 118
    plot_height = height - 150
    angle_bounds = _angle_bounds(case_result.views)

    cv2.rectangle(canvas, (plot_x, plot_y), (plot_x + plot_width, plot_y + plot_height), (252, 253, 255), thickness=-1)
    cv2.rectangle(canvas, (plot_x, plot_y), (plot_x + plot_width, plot_y + plot_height), BORDER_COLOR, thickness=1)
    _draw_angle_grid(canvas, (plot_x, plot_y, plot_width, plot_height), angle_bounds)
    _draw_primary_support_lines(canvas, case_result, evidence_rows, (plot_x, plot_y, plot_width, plot_height), angle_bounds)

    for row in evidence_rows:
        view = row.summary.view_input
        point = _map_angle_to_point(view.rao_lao, view.cra_cau, (plot_x, plot_y, plot_width, plot_height), angle_bounds)
        color = _role_color(row.role)
        cv2.circle(canvas, point, 11, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, point, 12, WHITE_COLOR, thickness=2, lineType=cv2.LINE_AA)
        if row.role == "primary":
            cv2.circle(canvas, point, 18, PRIMARY_COLOR, thickness=2, lineType=cv2.LINE_AA)
        _put_text(canvas, _truncate_text(view.view_id, 95, 0.42, 1), (point[0] + 14, point[1] - 10), 0.42, TEXT_COLOR, 1)

    _draw_angle_map_legend(canvas, x + 24, y + height - 44)


def _draw_primary_support_lines(
    canvas: np.ndarray,
    case_result: MultiViewCaseResult,
    evidence_rows: list[_ViewEvidence],
    plot_rect: tuple[int, int, int, int],
    angle_bounds: tuple[float, float, float, float],
) -> None:
    final_lesion = case_result.final_case_lesion
    if final_lesion is None:
        return
    primary_view = _find_view(case_result.views, final_lesion.primary_view_id)
    if primary_view is None:
        return

    primary_point = _map_angle_to_point(primary_view.rao_lao, primary_view.cra_cau, plot_rect, angle_bounds)
    for row in evidence_rows:
        if row.role not in {"distinct_support", "similar_support"}:
            continue
        view = row.summary.view_input
        point = _map_angle_to_point(view.rao_lao, view.cra_cau, plot_rect, angle_bounds)
        line_color = DISTINCT_COLOR if row.role == "distinct_support" else SIMILAR_COLOR
        cv2.line(canvas, primary_point, point, line_color, thickness=2, lineType=cv2.LINE_AA)


def _draw_conclusion_panel(
    canvas: np.ndarray,
    case_result: MultiViewCaseResult,
    evidence_rows: list[_ViewEvidence],
    rect: tuple[int, int, int, int],
) -> None:
    x, y, width, height = rect
    _draw_panel(canvas, rect)
    _put_text(canvas, "Final Conclusion", (x + 24, y + 38), 0.72, TEXT_COLOR, 2)

    final_lesion = case_result.final_case_lesion
    current_y = y + 82
    if final_lesion is None:
        current_y = _draw_wrapped_text(
            canvas,
            "No view-level lesion candidate passed into a final case-level lesion.",
            x + 24,
            current_y,
            width - 48,
            0.54,
            TEXT_COLOR,
            line_height=28,
        )
    else:
        current_y = _draw_key_value(canvas, "Decision", final_lesion.severity.title(), x + 24, current_y, width - 48)
        current_y = _draw_key_value(canvas, "Fused score", f"{final_lesion.total_score:.3f}", x + 24, current_y, width - 48)
        current_y = _draw_key_value(canvas, "Confidence", f"{final_lesion.confidence.label.title()} ({final_lesion.confidence.score:.3f})", x + 24, current_y, width - 48)
        current_y = _draw_key_value(canvas, "Primary view", final_lesion.primary_view_id, x + 24, current_y, width - 48)
        current_y = _draw_key_value(canvas, "Cross-view support", f"{final_lesion.distinct_view_support_score:.3f}", x + 24, current_y, width - 48)

    ranked_rows = [row for row in evidence_rows if row.contribution > 0.0]
    ranked_rows.sort(key=lambda row: row.contribution, reverse=True)
    _put_text(canvas, "Top Contributing Views", (x + 24, current_y + 24), 0.58, TEXT_COLOR, 2)
    current_y += 58
    if not ranked_rows:
        _put_text(canvas, "None", (x + 24, current_y), 0.5, MUTED_TEXT_COLOR, 1)
    else:
        max_contribution = max(row.contribution for row in ranked_rows)
        for row in ranked_rows[:5]:
            current_y = _draw_contribution_line(canvas, row, x + 24, current_y, width - 48, max_contribution)

    distinct_count = sum(1 for row in evidence_rows if row.role == "distinct_support")
    similar_count = sum(1 for row in evidence_rows if row.role == "similar_support")
    similar_label = "Compatible projection support views" if _uses_projection_view_metadata(case_result) else "Similar-angle support views"
    summary = f"Distinct support views: {distinct_count}   {similar_label}: {similar_count}"
    _put_text(canvas, summary, (x + 24, y + height - 30), 0.46, MUTED_TEXT_COLOR, 1)


def _draw_view_table(
    canvas: np.ndarray,
    evidence_rows: list[_ViewEvidence],
    rect: tuple[int, int, int, int],
    *,
    uses_projection_metadata: bool,
) -> None:
    x, y, width, height = rect
    _draw_panel(canvas, rect)
    _put_text(canvas, "Per-View Evidence", (x + 24, y + 38), 0.72, TEXT_COLOR, 2)
    _put_text(
        canvas,
        "Contribution is the actual score component used for the final selected case lesion.",
        (x + 24, y + 66),
        0.44,
        MUTED_TEXT_COLOR,
        1,
    )

    header_y = y + 104
    columns = {
        "thumbnail": x + 24,
        "view": x + 168,
        "angle": x + 330,
        "candidate": x + 520,
        "score": x + 670,
        "weight": x + 820,
        "contribution": x + 980,
        "role": x + 1160,
    }
    for label, column_x in [
        ("View", columns["view"]),
        ("Projection" if uses_projection_metadata else "Angle", columns["angle"]),
        ("Candidate", columns["candidate"]),
        ("Score", columns["score"]),
        ("Weight", columns["weight"]),
        ("Contribution", columns["contribution"]),
        ("Role", columns["role"]),
    ]:
        _put_text(canvas, label, (column_x, header_y), 0.43, MUTED_TEXT_COLOR, 1)

    row_height = 98
    row_y = header_y + 18
    for index, row in enumerate(evidence_rows):
        if row_y + row_height > y + height - 10:
            _put_text(canvas, f"+ {len(evidence_rows) - index} more views", (x + 24, row_y + 34), 0.5, MUTED_TEXT_COLOR, 1)
            break

        if index % 2 == 0:
            cv2.rectangle(canvas, (x + 14, row_y), (x + width - 14, row_y + row_height - 8), ROW_ALT_COLOR, thickness=-1)

        _draw_thumbnail(canvas, row.summary, columns["thumbnail"], row_y + 12, 116, 70)
        _put_text(canvas, _truncate_text(row.summary.view_input.view_id, 130, 0.5, 1), (columns["view"], row_y + 34), 0.5, TEXT_COLOR, 1)
        _put_text(canvas, _truncate_text(row.summary.view_input.sequence_id, 130, 0.42, 1), (columns["view"], row_y + 60), 0.42, MUTED_TEXT_COLOR, 1)
        _put_text(
            canvas,
            _format_view_metadata(row.summary.view_input, uses_projection_metadata),
            (columns["angle"], row_y + 40),
            0.46,
            TEXT_COLOR,
            1,
        )
        _put_text(canvas, _format_candidate(row), (columns["candidate"], row_y + 40), 0.46, TEXT_COLOR, 1)
        _put_text(canvas, _format_optional_score(row.candidate_score), (columns["score"], row_y + 40), 0.46, TEXT_COLOR, 1)
        _put_text(canvas, _format_optional_score(row.diversity_weight), (columns["weight"], row_y + 40), 0.46, TEXT_COLOR, 1)
        _draw_score_bar(canvas, columns["contribution"], row_y + 24, 120, row.contribution, _role_color(row.role))
        _put_text(canvas, f"{row.contribution:.3f}", (columns["contribution"], row_y + 68), 0.43, TEXT_COLOR, 1)
        _draw_role_pill(canvas, row.role_label, columns["role"], row_y + 22, _role_color(row.role))
        row_y += row_height


def _draw_thumbnail(
    canvas: np.ndarray,
    summary: MultiViewPerViewSummary,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    thumbnail = _load_temporal_summary_thumbnail(summary.view_input.temporal_fusion_json_path, width, height)
    canvas[y : y + height, x : x + width] = thumbnail
    cv2.rectangle(canvas, (x, y), (x + width, y + height), BORDER_COLOR, thickness=1)


def _load_temporal_summary_thumbnail(temporal_json_path: Path, width: int, height: int) -> np.ndarray:
    summary_path = temporal_json_path.with_name(f"{temporal_json_path.stem}{TEMPORAL_SUMMARY_IMAGE_SUFFIX}")
    image = None if not summary_path.is_file() else cv2.imread(str(summary_path), cv2.IMREAD_COLOR)
    if image is None:
        placeholder = np.full((height, width, 3), (230, 234, 240), dtype=np.uint8)
        _put_text(placeholder, "No", (width // 2 - 18, height // 2 - 4), 0.44, MUTED_TEXT_COLOR, 1)
        _put_text(placeholder, "thumb", (width // 2 - 28, height // 2 + 18), 0.44, MUTED_TEXT_COLOR, 1)
        return placeholder

    return _fit_image_to_box(image, width, height)


def _fit_image_to_box(image: np.ndarray, width: int, height: int) -> np.ndarray:
    fitted = np.full((height, width, 3), (230, 234, 240), dtype=np.uint8)
    image_height, image_width = image.shape[:2]
    if image_height <= 0 or image_width <= 0:
        return fitted

    scale = min(width / image_width, height / image_height)
    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    offset_x = (width - resized_width) // 2
    offset_y = (height - resized_height) // 2
    fitted[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return fitted


def _draw_angle_grid(
    canvas: np.ndarray,
    plot_rect: tuple[int, int, int, int],
    bounds: tuple[float, float, float, float],
) -> None:
    plot_x, plot_y, plot_width, plot_height = plot_rect
    min_x, max_x, min_y, max_y = bounds
    for value in _tick_values(min_x, max_x):
        px = _map_angle_to_point(value, min_y, plot_rect, bounds)[0]
        cv2.line(canvas, (px, plot_y), (px, plot_y + plot_height), GRID_COLOR, 1, cv2.LINE_AA)
        _put_text(canvas, f"{value:g}", (px - 12, plot_y + plot_height + 24), 0.36, MUTED_TEXT_COLOR, 1)
    for value in _tick_values(min_y, max_y):
        py = _map_angle_to_point(min_x, value, plot_rect, bounds)[1]
        cv2.line(canvas, (plot_x, py), (plot_x + plot_width, py), GRID_COLOR, 1, cv2.LINE_AA)
        _put_text(canvas, f"{value:g}", (plot_x - 46, py + 5), 0.36, MUTED_TEXT_COLOR, 1)

    if min_x <= 0 <= max_x:
        px = _map_angle_to_point(0, min_y, plot_rect, bounds)[0]
        cv2.line(canvas, (px, plot_y), (px, plot_y + plot_height), (170, 181, 194), 1, cv2.LINE_AA)
    if min_y <= 0 <= max_y:
        py = _map_angle_to_point(min_x, 0, plot_rect, bounds)[1]
        cv2.line(canvas, (plot_x, py), (plot_x + plot_width, py), (170, 181, 194), 1, cv2.LINE_AA)


def _draw_angle_map_legend(canvas: np.ndarray, x: int, y: int) -> None:
    legend_items = [
        ("primary", PRIMARY_COLOR),
        ("distinct support", DISTINCT_COLOR),
        ("similar support", SIMILAR_COLOR),
        ("candidate", CANDIDATE_COLOR),
        ("none", NO_CANDIDATE_COLOR),
    ]
    current_x = x
    for label, color in legend_items:
        cv2.circle(canvas, (current_x + 8, y - 5), 6, color, thickness=-1, lineType=cv2.LINE_AA)
        _put_text(canvas, label, (current_x + 20, y), 0.38, MUTED_TEXT_COLOR, 1)
        current_x += _text_width(label, 0.38, 1) + 48


def _draw_matrix_legend(
    canvas: np.ndarray,
    x: int,
    y: int,
    uses_projection_metadata: bool,
    duplicate_threshold: float,
    distinct_threshold: float,
) -> None:
    if uses_projection_metadata:
        items = [
            ("no support", _weight_color(0.0)),
            ("same group", _weight_color(0.25)),
            ("LCA vs LCA2", _weight_color(0.65)),
        ]
    else:
        items = [
            (f"similar <= {duplicate_threshold:g} deg", _weight_color(0.25)),
            ("partial", _weight_color(0.6)),
            (f"distinct >= {distinct_threshold:g} deg", _weight_color(1.0)),
        ]
    current_x = x
    for label, color in items:
        cv2.rectangle(canvas, (current_x, y - 16), (current_x + 28, y + 4), color, thickness=-1)
        cv2.rectangle(canvas, (current_x, y - 16), (current_x + 28, y + 4), BORDER_COLOR, thickness=1)
        _put_text(canvas, label, (current_x + 38, y), 0.43, MUTED_TEXT_COLOR, 1)
        current_x += _text_width(label, 0.43, 1) + 74


def _draw_contribution_line(
    canvas: np.ndarray,
    row: _ViewEvidence,
    x: int,
    y: int,
    width: int,
    max_contribution: float,
) -> int:
    view_id = _truncate_text(row.summary.view_input.view_id, 210, 0.47, 1)
    _put_text(canvas, view_id, (x, y), 0.47, TEXT_COLOR, 1)
    bar_x = x + 230
    bar_width = width - 330
    normalized = 0.0 if max_contribution <= 0.0 else row.contribution / max_contribution
    cv2.rectangle(canvas, (bar_x, y - 14), (bar_x + bar_width, y + 4), (224, 230, 238), thickness=-1)
    cv2.rectangle(canvas, (bar_x, y - 14), (bar_x + int(bar_width * normalized), y + 4), _role_color(row.role), thickness=-1)
    _put_text(canvas, f"{row.contribution:.3f}", (bar_x + bar_width + 16, y), 0.43, TEXT_COLOR, 1)
    return y + 34


def _draw_score_bar(
    canvas: np.ndarray,
    x: int,
    y: int,
    width: int,
    score: float,
    color: tuple[int, int, int],
) -> None:
    clamped_score = max(0.0, min(1.0, float(score)))
    cv2.rectangle(canvas, (x, y), (x + width, y + 14), (224, 230, 238), thickness=-1)
    cv2.rectangle(canvas, (x, y), (x + int(width * clamped_score), y + 14), color, thickness=-1)
    cv2.rectangle(canvas, (x, y), (x + width, y + 14), BORDER_COLOR, thickness=1)


def _draw_role_pill(
    canvas: np.ndarray,
    label: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    width = min(260, max(90, _text_width(label, 0.42, 1) + 26))
    cv2.rectangle(canvas, (x, y), (x + width, y + 28), color, thickness=-1)
    _put_text(canvas, label, (x + 12, y + 19), 0.42, WHITE_COLOR, 1)


def _draw_key_value(
    canvas: np.ndarray,
    key: str,
    value: str,
    x: int,
    y: int,
    max_width: int,
) -> int:
    _put_text(canvas, key, (x, y), 0.42, MUTED_TEXT_COLOR, 1)
    return _draw_wrapped_text(canvas, value, x + 180, y, max_width - 180, 0.5, TEXT_COLOR, line_height=28) + 4


def _draw_panel(canvas: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    x, y, width, height = rect
    cv2.rectangle(canvas, (x, y), (x + width, y + height), PANEL_COLOR, thickness=-1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (43, 49, 58), thickness=1)


def _role_label(role: str, uses_projection_metadata: bool) -> str:
    if uses_projection_metadata:
        return {
            "primary": "primary lesion",
            "distinct_support": "distinct projection support",
            "similar_support": "compatible projection support",
            "candidate": "candidate only",
            "none": "no lesion candidate",
        }.get(role, role)
    return {
        "primary": "primary lesion",
        "distinct_support": "distinct-angle support",
        "similar_support": "similar-angle support",
        "candidate": "candidate only",
        "none": "no lesion candidate",
    }.get(role, role)


def _role_color(role: str) -> tuple[int, int, int]:
    return {
        "primary": PRIMARY_COLOR,
        "distinct_support": DISTINCT_COLOR,
        "similar_support": SIMILAR_COLOR,
        "candidate": CANDIDATE_COLOR,
        "none": NO_CANDIDATE_COLOR,
    }.get(role, NO_CANDIDATE_COLOR)


def _weight_color(weight: float) -> tuple[int, int, int]:
    if weight <= 0.0:
        return (218, 224, 232)
    clamped = max(0.25, min(1.0, float(weight)))
    normalized = (clamped - 0.25) / 0.75
    low = np.asarray(SIMILAR_COLOR, dtype=np.float64)
    high = np.asarray(DISTINCT_COLOR, dtype=np.float64)
    return tuple(np.rint(low + ((high - low) * normalized)).astype(np.uint8).tolist())


def _angle_bounds(views: list[MultiViewViewInput]) -> tuple[float, float, float, float]:
    if not views:
        return (-45.0, 45.0, -45.0, 45.0)

    rao_lao_values = [float(view.rao_lao) for view in views] + [0.0]
    cra_cau_values = [float(view.cra_cau) for view in views] + [0.0]
    min_x, max_x = min(rao_lao_values), max(rao_lao_values)
    min_y, max_y = min(cra_cau_values), max(cra_cau_values)
    x_padding = max(10.0, (max_x - min_x) * 0.20)
    y_padding = max(10.0, (max_y - min_y) * 0.20)
    return (
        float(np.floor((min_x - x_padding) / 10.0) * 10.0),
        float(np.ceil((max_x + x_padding) / 10.0) * 10.0),
        float(np.floor((min_y - y_padding) / 10.0) * 10.0),
        float(np.ceil((max_y + y_padding) / 10.0) * 10.0),
    )


def _map_angle_to_point(
    rao_lao: float,
    cra_cau: float,
    plot_rect: tuple[int, int, int, int],
    bounds: tuple[float, float, float, float],
) -> tuple[int, int]:
    plot_x, plot_y, plot_width, plot_height = plot_rect
    min_x, max_x, min_y, max_y = bounds
    x_position = plot_x + int(round((float(rao_lao) - min_x) / max(max_x - min_x, 1.0) * plot_width))
    y_position = plot_y + plot_height - int(round((float(cra_cau) - min_y) / max(max_y - min_y, 1.0) * plot_height))
    return x_position, y_position


def _tick_values(min_value: float, max_value: float) -> list[float]:
    span = max_value - min_value
    step = 10.0 if span <= 60.0 else 20.0
    first_tick = np.ceil(min_value / step) * step
    values: list[float] = []
    value = first_tick
    while value <= max_value + 0.001:
        values.append(float(value))
        value += step
    return values


def _find_view(views: list[MultiViewViewInput], view_id: str) -> MultiViewViewInput | None:
    return next((view for view in views if view.view_id == view_id), None)


def _format_angle(view: MultiViewViewInput) -> str:
    horizontal_label = "LAO" if view.rao_lao >= 0.0 else "RAO"
    vertical_label = "CRA" if view.cra_cau >= 0.0 else "CAU"
    return f"{horizontal_label} {abs(view.rao_lao):.1f} / {vertical_label} {abs(view.cra_cau):.1f}"


def _format_view_metadata(view: MultiViewViewInput, uses_projection_metadata: bool) -> str:
    if not uses_projection_metadata:
        return _format_angle(view)
    group = view.projection_group or "unknown"
    side = view.coronary_side or "unknown"
    if view.projection_status == "ambiguous" and view.projection_groups:
        group = "+".join(view.projection_groups)
    return f"{group} / {side}"


def _uses_projection_view_metadata(case_result: MultiViewCaseResult) -> bool:
    mode = case_result.fusion_metadata.config.view_diversity_mode
    if mode == "projection_group":
        return True
    if mode == "auto":
        return any(_view_has_projection_metadata(view) for view in case_result.views)
    return False


def _view_has_projection_metadata(view: MultiViewViewInput) -> bool:
    return (
        view.projection_group is not None
        or view.projection_groups is not None
        or view.coronary_side is not None
        or view.projection_status is not None
        or view.angle_status is not None
    )


def _format_candidate(row: _ViewEvidence) -> str:
    if row.summary.candidate_count <= 0:
        return "none"
    return f"{row.summary.candidate_count} candidate" if row.summary.candidate_count == 1 else f"{row.summary.candidate_count} candidates"


def _format_optional_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _draw_wrapped_text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    max_width: int,
    font_scale: float,
    color: tuple[int, int, int],
    *,
    thickness: int = 1,
    line_height: int = 26,
) -> int:
    current_y = y
    for line in _wrap_text(text, max_width=max_width, font_scale=font_scale, thickness=thickness):
        _put_text(canvas, line, (x, current_y), font_scale, color, thickness)
        current_y += line_height
    return current_y


def _wrap_text(text: str, *, max_width: int, font_scale: float, thickness: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current_line = words[0]
    for word in words[1:]:
        candidate_line = f"{current_line} {word}"
        if _text_width(candidate_line, font_scale, thickness) <= max_width:
            current_line = candidate_line
            continue
        lines.append(current_line)
        current_line = word
    lines.append(current_line)
    return lines


def _truncate_text(text: str, max_width: int, font_scale: float, thickness: int) -> str:
    if _text_width(text, font_scale, thickness) <= max_width:
        return text
    ellipsis = "..."
    truncated = text
    while truncated and _text_width(f"{truncated}{ellipsis}", font_scale, thickness) > max_width:
        truncated = truncated[:-1]
    return f"{truncated}{ellipsis}" if truncated else ellipsis


def _put_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def _text_width(text: str, font_scale: float, thickness: int) -> int:
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    return int(text_size[0])


__all__ = [
    "build_multiview_visualization_paths",
    "create_multiview_summary_visualization",
    "create_multiview_support_matrix_visualization",
    "save_multiview_visualization_outputs",
]
