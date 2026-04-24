from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MultiViewViewInput:
    view_id: str
    sequence_id: str
    rao_lao: float
    cra_cau: float
    temporal_fusion_json_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "sequence_id": self.sequence_id,
            "rao_lao": float(self.rao_lao),
            "cra_cau": float(self.cra_cau),
            "temporal_fusion_json": str(self.temporal_fusion_json_path),
        }


@dataclass(slots=True)
class MultiViewCaseInput:
    case_id: str
    views: list[MultiViewViewInput]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "views": [view.to_dict() for view in self.views],
        }


@dataclass(slots=True)
class ViewLevelLesionCandidate:
    lesion_id: int
    track_id: int
    severity: str
    supporting_frame_count: int
    total_frame_count: int
    persistence_ratio: float
    frame_indices: list[int]
    median_degree: float
    max_degree: float
    degree_std: float
    max_frame_gap: int
    median_centerline_position: float | None = None
    dominant_centerline_component_id: int | None = None

    @property
    def base_score(self) -> float:
        return float(self.median_degree * self.persistence_ratio)

    def to_dict(self) -> dict[str, Any]:
        positions_payload: dict[str, Any] = {}
        if self.median_centerline_position is not None:
            positions_payload["median_centerline_position"] = float(self.median_centerline_position)
        if self.dominant_centerline_component_id is not None:
            positions_payload["dominant_centerline_component_id"] = int(self.dominant_centerline_component_id)

        return {
            "lesion_id": int(self.lesion_id),
            "track_id": int(self.track_id),
            "severity": self.severity,
            "supporting_frame_count": int(self.supporting_frame_count),
            "total_frame_count": int(self.total_frame_count),
            "persistence_ratio": float(self.persistence_ratio),
            "frame_indices": list(self.frame_indices),
            "degrees": {
                "median": float(self.median_degree),
                "max": float(self.max_degree),
            },
            "stability": {
                "degree_std": float(self.degree_std),
                "max_frame_gap": int(self.max_frame_gap),
            },
            "positions": positions_payload,
        }


@dataclass(slots=True)
class LoadedMultiViewView:
    view_input: MultiViewViewInput
    final_lesion: ViewLevelLesionCandidate | None
    persistent_lesions: list[ViewLevelLesionCandidate]

    @property
    def candidate_source(self) -> str:
        if self.persistent_lesions:
            return "persistent_lesions"
        if self.final_lesion is not None:
            return "final_lesion"
        return "none"

    @property
    def candidates(self) -> list[ViewLevelLesionCandidate]:
        if self.persistent_lesions:
            return list(self.persistent_lesions)
        if self.final_lesion is not None:
            return [self.final_lesion]
        return []

    def to_dict(self) -> dict[str, Any]:
        return {
            "view": self.view_input.to_dict(),
            "candidate_source": self.candidate_source,
            "final_lesion": None if self.final_lesion is None else self.final_lesion.to_dict(),
            "persistent_lesions": [candidate.to_dict() for candidate in self.persistent_lesions],
        }


@dataclass(slots=True)
class LoadedMultiViewCase:
    case_id: str
    views: list[LoadedMultiViewView]

    @property
    def view_count(self) -> int:
        return len(self.views)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "views": [view.to_dict() for view in self.views],
        }


@dataclass(slots=True)
class LesionCandidateScore:
    base_score: float
    stability_adjustment: float
    candidate_score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "base_score": float(self.base_score),
            "stability_adjustment": float(self.stability_adjustment),
            "candidate_score": float(self.candidate_score),
        }


@dataclass(slots=True)
class ConfidenceSummary:
    score: float
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "label": self.label,
        }


@dataclass(slots=True)
class MultiViewFusionConfig:
    duplicate_view_angle_distance_degrees: float = 20.0
    distinct_view_angle_distance_degrees: float = 45.0
    support_score_scale: float = 0.20
    medium_confidence_threshold: float = 0.45
    high_confidence_threshold: float = 0.75

    def to_dict(self) -> dict[str, Any]:
        return {
            "duplicate_view_angle_distance_degrees": float(self.duplicate_view_angle_distance_degrees),
            "distinct_view_angle_distance_degrees": float(self.distinct_view_angle_distance_degrees),
            "support_score_scale": float(self.support_score_scale),
            "confidence_thresholds": {
                "medium": float(self.medium_confidence_threshold),
                "high": float(self.high_confidence_threshold),
            },
        }


@dataclass(slots=True)
class MultiViewFusionMetadata:
    selection_rule: str
    total_candidate_count: int
    config: MultiViewFusionConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_rule": self.selection_rule,
            "total_candidate_count": int(self.total_candidate_count),
            **self.config.to_dict(),
        }


@dataclass(slots=True)
class MultiViewPerViewSummary:
    view_input: MultiViewViewInput
    candidate_source: str
    candidate_count: int
    best_candidate: ViewLevelLesionCandidate | None
    best_candidate_score: LesionCandidateScore | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_input.view_id,
            "sequence_id": self.view_input.sequence_id,
            "rao_lao": float(self.view_input.rao_lao),
            "cra_cau": float(self.view_input.cra_cau),
            "candidate_source": self.candidate_source,
            "candidate_count": int(self.candidate_count),
            "best_candidate": None if self.best_candidate is None else self.best_candidate.to_dict(),
            "best_candidate_score": None if self.best_candidate_score is None else self.best_candidate_score.to_dict(),
        }


@dataclass(slots=True)
class FinalCaseLesion:
    primary_view_id: str
    primary_sequence_id: str
    lesion: ViewLevelLesionCandidate
    score_breakdown: LesionCandidateScore
    distinct_view_support_score: float
    total_score: float
    supporting_view_ids: list[str]
    distinct_supporting_view_ids: list[str]
    confidence: ConfidenceSummary

    @property
    def severity(self) -> str:
        return self.lesion.severity

    @property
    def median_degree(self) -> float:
        return float(self.lesion.median_degree)

    def to_dict(self) -> dict[str, Any]:
        positions_payload: dict[str, Any] = {}
        if self.lesion.median_centerline_position is not None:
            positions_payload["median_centerline_position"] = float(self.lesion.median_centerline_position)
        if self.lesion.dominant_centerline_component_id is not None:
            positions_payload["dominant_centerline_component_id"] = int(self.lesion.dominant_centerline_component_id)

        return {
            "primary_view_id": self.primary_view_id,
            "primary_sequence_id": self.primary_sequence_id,
            "lesion_id": int(self.lesion.lesion_id),
            "track_id": int(self.lesion.track_id),
            "severity": self.lesion.severity,
            "supporting_frame_count": int(self.lesion.supporting_frame_count),
            "total_frame_count": int(self.lesion.total_frame_count),
            "persistence_ratio": float(self.lesion.persistence_ratio),
            "frame_indices": list(self.lesion.frame_indices),
            "degrees": {
                "median": float(self.lesion.median_degree),
                "max": float(self.lesion.max_degree),
            },
            "stability": {
                "degree_std": float(self.lesion.degree_std),
                "max_frame_gap": int(self.lesion.max_frame_gap),
            },
            "positions": positions_payload,
            "score_breakdown": self.score_breakdown.to_dict(),
            "distinct_view_support_score": float(self.distinct_view_support_score),
            "total_score": float(self.total_score),
            "confidence": self.confidence.to_dict(),
            "supporting_view_ids": list(self.supporting_view_ids),
            "distinct_supporting_view_ids": list(self.distinct_supporting_view_ids),
        }


@dataclass(slots=True)
class MultiViewCaseResult:
    case_id: str
    views: list[MultiViewViewInput]
    per_view_summary: list[MultiViewPerViewSummary]
    final_case_lesion: FinalCaseLesion | None
    confidence: ConfidenceSummary
    supporting_views: list[str]
    fusion_metadata: MultiViewFusionMetadata

    @property
    def view_count(self) -> int:
        return len(self.views)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "view_count": self.view_count,
            "views": [view.to_dict() for view in self.views],
            "per_view_summary": [summary.to_dict() for summary in self.per_view_summary],
            "final_case_lesion": None if self.final_case_lesion is None else self.final_case_lesion.to_dict(),
            "confidence": self.confidence.to_dict(),
            "supporting_views": list(self.supporting_views),
            "fusion_metadata": self.fusion_metadata.to_dict(),
        }
