from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any

from angio_keyframes.cadica import read_selected_frames
from angio_keyframes.images import is_supported_image_name


INTEGER_RE = re.compile(r"\d+")


@dataclass(frozen=True, slots=True)
class CadicaKeyframeReference:
    patient_id: str
    video_id: str
    selected_frame_ids: set[int]
    selected_frames_path: Path


@dataclass(frozen=True, slots=True)
class CadicaKeyframeBenchmarkResult:
    frame_rows: list[dict[str, Any]]
    video_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    output_paths: dict[str, Path]


def run_cadica_keyframe_benchmark(
    *,
    cadica_root: str | Path,
    extracted_keyframes_root: str | Path,
    output_root: str | Path,
) -> CadicaKeyframeBenchmarkResult:
    resolved_cadica_root = Path(cadica_root)
    resolved_extracted_root = Path(extracted_keyframes_root)
    resolved_output_root = Path(output_root)
    _validate_inputs(resolved_cadica_root, resolved_extracted_root)

    references = load_cadica_keyframe_references(resolved_cadica_root)
    frame_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    for reference in references:
        predicted_frame_ids = _read_extracted_frame_ids(
            resolved_extracted_root / reference.patient_id / reference.video_id
        )
        video_frame_rows, video_row = _evaluate_video(reference, predicted_frame_ids)
        frame_rows.extend(video_frame_rows)
        video_rows.append(video_row)

    summary = _build_summary(
        video_rows,
        cadica_root=resolved_cadica_root,
        extracted_keyframes_root=resolved_extracted_root,
        output_root=resolved_output_root,
    )
    output_paths = save_cadica_keyframe_benchmark_outputs(
        output_root=resolved_output_root,
        frame_rows=frame_rows,
        video_rows=video_rows,
        summary=summary,
    )
    return CadicaKeyframeBenchmarkResult(
        frame_rows=frame_rows,
        video_rows=video_rows,
        summary=summary,
        output_paths=output_paths,
    )


def load_cadica_keyframe_references(cadica_root: str | Path) -> list[CadicaKeyframeReference]:
    selected_root = Path(cadica_root) / "selectedVideos"
    if not selected_root.exists():
        raise FileNotFoundError(f"CADICA selectedVideos directory does not exist: {selected_root}")
    if not selected_root.is_dir():
        raise NotADirectoryError(f"CADICA selectedVideos path is not a directory: {selected_root}")

    references: list[CadicaKeyframeReference] = []
    for patient_dir in _sorted_directories(selected_root):
        for video_dir in _sorted_directories(patient_dir):
            selected_frames_path = video_dir / f"{patient_dir.name}_{video_dir.name}_selectedFrames.txt"
            if not selected_frames_path.is_file():
                continue
            references.append(
                CadicaKeyframeReference(
                    patient_id=patient_dir.name,
                    video_id=video_dir.name,
                    selected_frame_ids=read_selected_frames(selected_frames_path),
                    selected_frames_path=selected_frames_path,
                )
            )
    if not references:
        raise FileNotFoundError(f"No CADICA selectedFrames files were found under: {selected_root}")
    return references


def save_cadica_keyframe_benchmark_outputs(
    *,
    output_root: str | Path,
    frame_rows: list[dict[str, Any]],
    video_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Path]:
    resolved_output_root = Path(output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "frame_rows_csv": resolved_output_root / "cadica_keyframe_frame_rows.csv",
        "frame_rows_jsonl": resolved_output_root / "cadica_keyframe_frame_rows.jsonl",
        "video_rows_csv": resolved_output_root / "cadica_keyframe_video_rows.csv",
        "summary_json": resolved_output_root / "cadica_keyframe_summary.json",
        "missed_keyframes_csv": resolved_output_root / "missed_keyframes.csv",
        "extra_keyframes_csv": resolved_output_root / "extra_keyframes.csv",
    }
    _write_csv(output_paths["frame_rows_csv"], frame_rows, fieldnames=FRAME_ROW_FIELDS)
    _write_jsonl(output_paths["frame_rows_jsonl"], frame_rows)
    _write_csv(output_paths["video_rows_csv"], video_rows, fieldnames=VIDEO_ROW_FIELDS)
    _write_json(output_paths["summary_json"], summary)
    _write_csv(
        output_paths["missed_keyframes_csv"],
        [row for row in frame_rows if row["outcome"] == "FN"],
        fieldnames=FRAME_ROW_FIELDS,
    )
    _write_csv(
        output_paths["extra_keyframes_csv"],
        [row for row in frame_rows if row["outcome"] == "FP"],
        fieldnames=FRAME_ROW_FIELDS,
    )
    return output_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark extracted CADICA keyframes against CADICA selectedFrames references.",
    )
    parser.add_argument("--cadica-root", required=True, help="Path to the CADICA dataset root.")
    parser.add_argument(
        "--extracted-keyframes-root",
        required=True,
        help="Root containing extracted keyframes as pX/vY/*.png.",
    )
    parser.add_argument("--output-root", required=True, help="Directory where benchmark outputs will be written.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_cadica_keyframe_benchmark(
            cadica_root=args.cadica_root,
            extracted_keyframes_root=args.extracted_keyframes_root,
            output_root=args.output_root,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as exc:
        print(f"CADICA keyframe benchmark failed: {exc}", file=sys.stderr)
        return 2

    metrics = result.summary["metrics"]
    print(f"CADICA keyframe video rows: {result.output_paths['video_rows_csv']}")
    print(f"Evaluated videos: {metrics['video_count']}")
    print(f"Frame precision: {_format_optional_float(metrics['precision'])}")
    print(f"Frame recall: {_format_optional_float(metrics['recall'])}")
    print(f"Summary: {result.output_paths['summary_json']}")
    return 0


def _evaluate_video(
    reference: CadicaKeyframeReference,
    predicted_frame_ids: set[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_frame_ids = set(reference.selected_frame_ids)
    matched_frame_ids = selected_frame_ids & predicted_frame_ids
    missed_frame_ids = selected_frame_ids - predicted_frame_ids
    extra_frame_ids = predicted_frame_ids - selected_frame_ids
    frame_rows = [
        _frame_row(reference, frame_id=frame_id, label_selected=True, predicted_selected=True, outcome="TP")
        for frame_id in sorted(matched_frame_ids)
    ]
    frame_rows.extend(
        _frame_row(reference, frame_id=frame_id, label_selected=True, predicted_selected=False, outcome="FN")
        for frame_id in sorted(missed_frame_ids)
    )
    frame_rows.extend(
        _frame_row(reference, frame_id=frame_id, label_selected=False, predicted_selected=True, outcome="FP")
        for frame_id in sorted(extra_frame_ids)
    )
    precision = _safe_divide(len(matched_frame_ids), len(predicted_frame_ids))
    recall = _safe_divide(len(matched_frame_ids), len(selected_frame_ids))
    f1 = _f1(precision, recall)
    video_row = {
        "patient_id": reference.patient_id,
        "video_id": reference.video_id,
        "gt_selected_count": len(selected_frame_ids),
        "predicted_count": len(predicted_frame_ids),
        "matched_count": len(matched_frame_ids),
        "missed_count": len(missed_frame_ids),
        "extra_count": len(extra_frame_ids),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": selected_frame_ids == predicted_frame_ids,
        "gt_frame_ids": sorted(selected_frame_ids),
        "predicted_frame_ids": sorted(predicted_frame_ids),
        "matched_frame_ids": sorted(matched_frame_ids),
        "missed_frame_ids": sorted(missed_frame_ids),
        "extra_frame_ids": sorted(extra_frame_ids),
        "selected_frames_path": str(reference.selected_frames_path),
    }
    return frame_rows, video_row


def _frame_row(
    reference: CadicaKeyframeReference,
    *,
    frame_id: int,
    label_selected: bool,
    predicted_selected: bool,
    outcome: str,
) -> dict[str, Any]:
    return {
        "patient_id": reference.patient_id,
        "video_id": reference.video_id,
        "frame_id": frame_id,
        "label_selected": label_selected,
        "predicted_selected": predicted_selected,
        "outcome": outcome,
        "selected_frames_path": str(reference.selected_frames_path),
    }


def _build_summary(
    video_rows: list[dict[str, Any]],
    *,
    cadica_root: Path,
    extracted_keyframes_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    total_gt = sum(int(row["gt_selected_count"]) for row in video_rows)
    total_predicted = sum(int(row["predicted_count"]) for row in video_rows)
    total_matched = sum(int(row["matched_count"]) for row in video_rows)
    total_missed = sum(int(row["missed_count"]) for row in video_rows)
    total_extra = sum(int(row["extra_count"]) for row in video_rows)
    precision = _safe_divide(total_matched, total_predicted)
    recall = _safe_divide(total_matched, total_gt)
    return {
        "note": "This benchmark compares extracted CADICA keyframe frame IDs against CADICA selectedFrames references.",
        "config": {
            "cadica_root": str(cadica_root),
            "extracted_keyframes_root": str(extracted_keyframes_root),
            "output_root": str(output_root),
        },
        "metrics": {
            "video_count": len(video_rows),
            "exact_match_videos": sum(1 for row in video_rows if row["exact_match"]),
            "total_gt_selected_frames": total_gt,
            "total_predicted_keyframes": total_predicted,
            "matched_frames": total_matched,
            "missed_frames": total_missed,
            "extra_frames": total_extra,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        },
        "counts": {
            "predicted_count_distribution": dict(sorted(Counter(int(row["predicted_count"]) for row in video_rows).items())),
            "gt_selected_count_distribution": dict(
                sorted(Counter(int(row["gt_selected_count"]) for row in video_rows).items())
            ),
        },
    }


def _read_extracted_frame_ids(video_output_dir: Path) -> set[int]:
    if not video_output_dir.is_dir():
        return set()
    frame_ids: set[int] = set()
    for image_path in sorted(video_output_dir.iterdir(), key=lambda path: _natural_sort_key(path.name)):
        if image_path.is_file() and is_supported_image_name(image_path.name):
            frame_ids.add(_frame_id_from_name(image_path.name))
    return frame_ids


def _frame_id_from_name(name: str) -> int:
    integer_matches = INTEGER_RE.findall(name)
    if not integer_matches:
        raise ValueError(f"Could not extract frame ID from filename: {name}")
    return int(integer_matches[-1])


def _sorted_directories(path: Path) -> list[Path]:
    return sorted((child for child in path.iterdir() if child.is_dir()), key=lambda child: _natural_sort_key(child.name))


def _natural_sort_key(value: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0.0:
        return None
    return float(2.0 * precision * recall / (precision + recall))


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field_name: _csv_value(row.get(field_name)) for field_name in fieldnames})


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def _format_optional_float(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _validate_inputs(cadica_root: Path, extracted_keyframes_root: Path) -> None:
    if not cadica_root.exists():
        raise FileNotFoundError(f"CADICA root does not exist: {cadica_root}")
    if not cadica_root.is_dir():
        raise NotADirectoryError(f"CADICA root is not a directory: {cadica_root}")
    if not extracted_keyframes_root.exists():
        raise FileNotFoundError(f"Extracted keyframes root does not exist: {extracted_keyframes_root}")
    if not extracted_keyframes_root.is_dir():
        raise NotADirectoryError(f"Extracted keyframes root is not a directory: {extracted_keyframes_root}")


FRAME_ROW_FIELDS = [
    "patient_id",
    "video_id",
    "frame_id",
    "label_selected",
    "predicted_selected",
    "outcome",
    "selected_frames_path",
]

VIDEO_ROW_FIELDS = [
    "patient_id",
    "video_id",
    "gt_selected_count",
    "predicted_count",
    "matched_count",
    "missed_count",
    "extra_count",
    "precision",
    "recall",
    "f1",
    "exact_match",
    "gt_frame_ids",
    "predicted_frame_ids",
    "matched_frame_ids",
    "missed_frame_ids",
    "extra_frame_ids",
    "selected_frames_path",
]


if __name__ == "__main__":
    raise SystemExit(main())
