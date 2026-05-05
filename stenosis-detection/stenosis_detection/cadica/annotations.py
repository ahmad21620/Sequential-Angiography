from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(slots=True)
class CadicaBox:
    patient_id: str
    video_id: str
    frame_id: int
    x: float
    y: float
    w: float
    h: float
    category: str | None
    source_path: Path


@dataclass(slots=True)
class CadicaFrame:
    patient_id: str
    video_id: str
    frame_id: int
    original_image_path: Path
    original_image_name: str
    prepared_image_name: str
    prepared_image_stem: str
    is_selected_frame: bool
    video_label: str
    frame_label: str
    boxes: list[CadicaBox]


FRAME_SCOPE_CHOICES = {"selected", "all_selected_videos"}
NEGATIVE_FRAME_SCOPE_CHOICES = {"selected", "all"}
_INTEGER_RE = re.compile(r"\d+")
_TOKEN_SPLIT_RE = re.compile(r"[,;\s]+")


def load_cadica_annotations(
    cadica_root: str | Path,
    *,
    frame_scope: str = "all_selected_videos",
    negative_frame_scope: str = "selected",
) -> list[CadicaFrame]:
    """Load frame-level CADICA labels from selectedVideos only."""
    if frame_scope not in FRAME_SCOPE_CHOICES:
        raise ValueError(f"frame_scope must be one of {sorted(FRAME_SCOPE_CHOICES)}, got {frame_scope!r}.")
    if negative_frame_scope not in NEGATIVE_FRAME_SCOPE_CHOICES:
        raise ValueError(
            "negative_frame_scope must be one of "
            f"{sorted(NEGATIVE_FRAME_SCOPE_CHOICES)}, got {negative_frame_scope!r}."
        )

    selected_root = Path(cadica_root) / "selectedVideos"
    if not selected_root.exists():
        raise FileNotFoundError(f"CADICA selectedVideos directory does not exist: {selected_root}")
    if not selected_root.is_dir():
        raise NotADirectoryError(f"CADICA selectedVideos path is not a directory: {selected_root}")

    frames: list[CadicaFrame] = []
    for patient_dir in _sorted_directories(selected_root):
        patient_id = patient_dir.name
        lesion_videos = read_video_list(patient_dir / "lesionVideos.txt")
        nonlesion_videos = read_video_list(patient_dir / "nonlesionVideos.txt")

        for video_dir in _sorted_directories(patient_dir):
            video_id = video_dir.name
            video_label = _video_label(video_id, lesion_videos=lesion_videos, nonlesion_videos=nonlesion_videos)
            selected_frame_ids = read_selected_frames(video_dir / f"{patient_id}_{video_id}_selectedFrames.txt")
            boxes_by_frame = read_groundtruth_boxes(video_dir / "groundtruth", patient_id, video_id)
            input_dir = video_dir / "input"
            if not input_dir.is_dir():
                continue

            image_paths = _selected_image_paths(input_dir, selected_frame_ids, frame_scope=frame_scope)
            for image_path in image_paths:
                frame_id = _frame_id_from_name(image_path.name)
                is_selected_frame = frame_id in selected_frame_ids
                boxes = boxes_by_frame.get(frame_id, [])
                prepared_image_name = f"slice_{frame_id:05d}.png"
                frames.append(
                    CadicaFrame(
                        patient_id=patient_id,
                        video_id=video_id,
                        frame_id=frame_id,
                        original_image_path=image_path.resolve(),
                        original_image_name=image_path.name,
                        prepared_image_name=prepared_image_name,
                        prepared_image_stem=Path(prepared_image_name).stem,
                        is_selected_frame=is_selected_frame,
                        video_label=video_label,
                        frame_label=_frame_label(
                            video_label=video_label,
                            has_boxes=bool(boxes),
                            is_selected_frame=is_selected_frame,
                            negative_frame_scope=negative_frame_scope,
                        ),
                        boxes=boxes,
                    )
                )

    return frames


def read_video_list(path: Path) -> set[str]:
    if not path.is_file():
        return set()

    video_ids: set[str] = set()
    for token in _tokens(path.read_text(encoding="utf-8-sig")):
        video_id = _video_id_from_token(token)
        if video_id is not None:
            video_ids.add(video_id)
    return video_ids


def read_selected_frames(path: Path) -> set[int]:
    if not path.is_file():
        return set()

    frame_ids: set[int] = set()
    for token in _tokens(path.read_text(encoding="utf-8-sig")):
        integer_matches = _INTEGER_RE.findall(token)
        if integer_matches:
            frame_ids.add(int(integer_matches[-1]))
    return frame_ids


def read_groundtruth_boxes(gt_dir: Path, patient_id: str, video_id: str) -> dict[int, list[CadicaBox]]:
    if not gt_dir.is_dir():
        return {}

    boxes_by_frame: dict[int, list[CadicaBox]] = {}
    for gt_path in sorted(gt_dir.glob("*.txt"), key=lambda path: _natural_sort_key(path.name)):
        frame_id = _frame_id_from_name(gt_path.name)
        for line_number, raw_line in enumerate(gt_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            box = _parse_groundtruth_line(
                line,
                patient_id=patient_id,
                video_id=video_id,
                frame_id=frame_id,
                source_path=gt_path.resolve(),
                line_number=line_number,
            )
            boxes_by_frame.setdefault(frame_id, []).append(box)
    return boxes_by_frame


def _selected_image_paths(input_dir: Path, selected_frame_ids: set[int], *, frame_scope: str) -> list[Path]:
    image_paths = sorted(input_dir.glob("*.png"), key=lambda path: _natural_sort_key(path.name))
    if frame_scope == "selected":
        return [path for path in image_paths if _frame_id_from_name(path.name) in selected_frame_ids]
    return image_paths


def _parse_groundtruth_line(
    line: str,
    *,
    patient_id: str,
    video_id: str,
    frame_id: int,
    source_path: Path,
    line_number: int,
) -> CadicaBox:
    fields = [field for field in _TOKEN_SPLIT_RE.split(line) if field]
    if len(fields) < 4:
        raise ValueError(f"{source_path}:{line_number}: expected at least x, y, w, h in GT row: {line!r}")

    try:
        x, y, w, h = (_float_field(field) for field in fields[:4])
    except ValueError as exc:
        raise ValueError(f"{source_path}:{line_number}: could not parse GT box row: {line!r}") from exc

    category = " ".join(fields[4:]).strip() if len(fields) > 4 else None
    return CadicaBox(
        patient_id=patient_id,
        video_id=video_id,
        frame_id=frame_id,
        x=x,
        y=y,
        w=w,
        h=h,
        category=category or None,
        source_path=source_path,
    )


def _float_field(value: str) -> float:
    return float(value.strip("[]()"))


def _frame_label(
    *,
    video_label: str,
    has_boxes: bool,
    is_selected_frame: bool,
    negative_frame_scope: str,
) -> str:
    if has_boxes:
        return "positive"
    if video_label == "nonlesion" and (negative_frame_scope == "all" or is_selected_frame):
        return "negative"
    return "unknown"


def _video_label(video_id: str, *, lesion_videos: set[str], nonlesion_videos: set[str]) -> str:
    if video_id in lesion_videos:
        return "lesion"
    if video_id in nonlesion_videos:
        return "nonlesion"
    return "unknown"


def _video_id_from_token(token: str) -> str | None:
    match = re.search(r"v(\d+)", token, flags=re.IGNORECASE)
    if match is not None:
        return f"v{int(match.group(1))}"
    if token.isdigit():
        return f"v{int(token)}"
    return None


def _frame_id_from_name(name: str) -> int:
    integer_matches = _INTEGER_RE.findall(name)
    if not integer_matches:
        raise ValueError(f"Could not extract CADICA frame ID from filename: {name}")
    return int(integer_matches[-1])


def _tokens(text: str) -> list[str]:
    return [token.strip().strip('"\'') for token in _TOKEN_SPLIT_RE.split(text) if token.strip()]


def _sorted_directories(path: Path) -> list[Path]:
    return sorted((child for child in path.iterdir() if child.is_dir()), key=lambda child: _natural_sort_key(child.name))


def _natural_sort_key(value: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]
