from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

from .annotations import CadicaBox, CadicaFrame, load_cadica_annotations


COPY_MODE_CHOICES = {"copy", "symlink", "hardlink"}
MANIFEST_FIELDS = [
    "patient_id",
    "video_id",
    "frame_id",
    "original_image_path",
    "original_image_name",
    "prepared_relative_path",
    "prepared_image_name",
    "is_selected_frame",
    "video_label",
    "frame_label",
    "box_count",
    "gt_box_source_paths",
    "gt_boxes_json",
]


def prepare_cadica_for_pipeline(
    cadica_root: str | Path,
    output_root: str | Path,
    *,
    frame_scope: str = "all_selected_videos",
    negative_frame_scope: str = "selected",
    copy_mode: str = "copy",
) -> dict[str, Path]:
    if copy_mode not in COPY_MODE_CHOICES:
        raise ValueError(f"copy_mode must be one of {sorted(COPY_MODE_CHOICES)}, got {copy_mode!r}.")

    resolved_output_root = Path(output_root)
    keyframes_root = resolved_output_root / "keyframes"
    frames = load_cadica_annotations(
        cadica_root,
        frame_scope=frame_scope,
        negative_frame_scope=negative_frame_scope,
    )

    rows: list[dict[str, Any]] = []
    frames_by_patient_video: dict[tuple[str, str], list[CadicaFrame]] = defaultdict(list)
    for frame in frames:
        prepared_relative_path = Path("keyframes") / frame.patient_id / frame.video_id / frame.prepared_image_name
        prepared_path = resolved_output_root / prepared_relative_path
        _materialize_image(frame.original_image_path, prepared_path, copy_mode=copy_mode)
        rows.append(_manifest_row(frame, prepared_relative_path))
        frames_by_patient_video[(frame.patient_id, frame.video_id)].append(frame)

    manifest_csv = resolved_output_root / "manifest.csv"
    manifest_jsonl = resolved_output_root / "manifest.jsonl"
    summary_json = resolved_output_root / "preparation_summary.json"
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(manifest_csv, rows)
    _write_jsonl(manifest_jsonl, rows)
    _write_patient_files(keyframes_root, frames_by_patient_video)
    _write_summary(
        summary_json,
        frames,
        cadica_root=Path(cadica_root),
        output_root=resolved_output_root,
        frame_scope=frame_scope,
        negative_frame_scope=negative_frame_scope,
        copy_mode=copy_mode,
    )

    return {
        "keyframes_root": keyframes_root,
        "manifest_csv": manifest_csv,
        "manifest_jsonl": manifest_jsonl,
        "preparation_summary_json": summary_json,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare CADICA selectedVideos for the sequential angiography pipeline.")
    parser.add_argument("--cadica-root", required=True, help="Path to the CADICA dataset root.")
    parser.add_argument("--output-root", required=True, help="Directory where prepared CADICA files will be written.")
    parser.add_argument(
        "--frame-scope",
        choices=["selected", "all_selected_videos"],
        default="all_selected_videos",
        help="Which frames from selectedVideos to prepare.",
    )
    parser.add_argument(
        "--negative-frame-scope",
        choices=["selected", "all"],
        default="selected",
        help="Which frames from nonlesion videos should be labeled negative.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=sorted(COPY_MODE_CHOICES),
        default="copy",
        help="How prepared image files should reference the original CADICA frames.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        outputs = prepare_cadica_for_pipeline(
            args.cadica_root,
            args.output_root,
            frame_scope=args.frame_scope,
            negative_frame_scope=args.negative_frame_scope,
            copy_mode=args.copy_mode,
        )
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        print(f"CADICA preparation failed: {exc}", file=sys.stderr)
        return 2

    print(f"Prepared CADICA keyframes: {outputs['keyframes_root']}")
    print(f"Manifest CSV: {outputs['manifest_csv']}")
    print(f"Manifest JSONL: {outputs['manifest_jsonl']}")
    print(f"Summary: {outputs['preparation_summary_json']}")
    return 0


def _materialize_image(source_path: Path, prepared_path: Path, *, copy_mode: str) -> None:
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    if prepared_path.exists() or prepared_path.is_symlink():
        if prepared_path.is_dir():
            raise IsADirectoryError(f"Prepared image path is a directory: {prepared_path}")
        prepared_path.unlink()

    if copy_mode == "copy":
        shutil.copy2(source_path, prepared_path)
    elif copy_mode == "symlink":
        prepared_path.symlink_to(source_path)
    elif copy_mode == "hardlink":
        os.link(source_path, prepared_path)
    else:  # pragma: no cover - validated by public entry point.
        raise ValueError(f"Unsupported copy mode: {copy_mode}")


def _manifest_row(frame: CadicaFrame, prepared_relative_path: Path) -> dict[str, Any]:
    return {
        "patient_id": frame.patient_id,
        "video_id": frame.video_id,
        "frame_id": frame.frame_id,
        "original_image_path": str(frame.original_image_path),
        "original_image_name": frame.original_image_name,
        "prepared_relative_path": prepared_relative_path.as_posix(),
        "prepared_image_name": frame.prepared_image_name,
        "is_selected_frame": frame.is_selected_frame,
        "video_label": frame.video_label,
        "frame_label": frame.frame_label,
        "box_count": len(frame.boxes),
        "gt_box_source_paths": sorted({str(box.source_path) for box in frame.boxes}),
        "gt_boxes_json": json.dumps([_box_to_dict(box) for box in frame.boxes], sort_keys=True),
    }


def _box_to_dict(box: CadicaBox) -> dict[str, Any]:
    return {
        "patient_id": box.patient_id,
        "video_id": box.video_id,
        "frame_id": box.frame_id,
        "x": box.x,
        "y": box.y,
        "w": box.w,
        "h": box.h,
        "category": box.category,
        "source_path": str(box.source_path),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in MANIFEST_FIELDS})


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_patient_files(keyframes_root: Path, frames_by_patient_video: dict[tuple[str, str], list[CadicaFrame]]) -> None:
    videos_by_patient: dict[str, dict[str, list[CadicaFrame]]] = defaultdict(dict)
    for (patient_id, video_id), frames in frames_by_patient_video.items():
        videos_by_patient[patient_id][video_id] = sorted(frames, key=lambda frame: frame.frame_id)

    for patient_id, videos in sorted(videos_by_patient.items()):
        patient_dir = keyframes_root / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        video_summaries = [_patient_video_summary(video_id, frames) for video_id, frames in sorted(videos.items())]
        patient_payload = {
            "patient_id": patient_id,
            "source_dataset": "CADICA",
            "video_count": len(video_summaries),
            "videos": video_summaries,
        }
        views_payload = {
            "case_id": patient_id,
            "metadata": {
                "source_dataset": "CADICA",
                "projection_angles_placeholder": True,
                "note": "Projection angles are placeholders until CADICAprojections.json support is implemented.",
            },
            "views": [
                {
                    "view_id": video["video_id"],
                    "sequence_id": video["video_id"],
                    "rao_lao": 0.0,
                    "cra_cau": 0.0,
                    "temporal_fusion_json": (
                        f"../../stenosis_temporal_results/{patient_id}/{video['video_id']}/view_temporal_fusion.json"
                    ),
                    "placeholder_projection_angles": True,
                }
                for video in video_summaries
            ],
        }
        (patient_dir / "patient.json").write_text(json.dumps(patient_payload, indent=2, sort_keys=True), encoding="utf-8")
        (patient_dir / "views.json").write_text(json.dumps(views_payload, indent=2, sort_keys=True), encoding="utf-8")


def _patient_video_summary(video_id: str, frames: list[CadicaFrame]) -> dict[str, Any]:
    frame_label_counts = Counter(frame.frame_label for frame in frames)
    return {
        "video_id": video_id,
        "video_label": frames[0].video_label if frames else "unknown",
        "frame_count": len(frames),
        "selected_frame_count": sum(1 for frame in frames if frame.is_selected_frame),
        "frame_label_counts": dict(sorted(frame_label_counts.items())),
    }


def _write_summary(
    path: Path,
    frames: list[CadicaFrame],
    *,
    cadica_root: Path,
    output_root: Path,
    frame_scope: str,
    negative_frame_scope: str,
    copy_mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    patients = {frame.patient_id for frame in frames}
    videos = {(frame.patient_id, frame.video_id) for frame in frames}
    payload = {
        "cadica_root": str(cadica_root),
        "output_root": str(output_root),
        "frame_scope": frame_scope,
        "negative_frame_scope": negative_frame_scope,
        "copy_mode": copy_mode,
        "patient_count": len(patients),
        "video_count": len(videos),
        "frame_count": len(frames),
        "selected_frame_count": sum(1 for frame in frames if frame.is_selected_frame),
        "box_count": sum(len(frame.boxes) for frame in frames),
        "video_label_counts": dict(sorted(Counter(frame.video_label for frame in frames).items())),
        "frame_label_counts": dict(sorted(Counter(frame.frame_label for frame in frames).items())),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _csv_value(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value
