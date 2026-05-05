from __future__ import annotations

from pathlib import Path
import re


CADICA_INPUT_FRAMES_DIRNAME = "input"
CADICA_SELECTED_VIDEOS_DIRNAME = "selectedVideos"
INTEGER_RE = re.compile(r"\d+")
TOKEN_SPLIT_RE = re.compile(r"[,;\s]+")


def resolve_cadica_selected_videos_root(input_path: Path) -> Path:
    selected_videos_root = input_path / CADICA_SELECTED_VIDEOS_DIRNAME
    if selected_videos_root.exists():
        if not selected_videos_root.is_dir():
            raise NotADirectoryError(
                f"CADICA selectedVideos path is not a directory: {selected_videos_root}"
            )
        return selected_videos_root
    return input_path


def cadica_selected_frames_path(frames_dir: Path, frames_dirname: str = CADICA_INPUT_FRAMES_DIRNAME) -> Path:
    frames_dirnames = {frames_dirname, CADICA_INPUT_FRAMES_DIRNAME}
    video_dir = frames_dir.parent if frames_dir.name in frames_dirnames else frames_dir
    return video_dir / f"{video_dir.parent.name}_{video_dir.name}_selectedFrames.txt"


def read_selected_frames(path: Path) -> set[int]:
    frame_ids: set[int] = set()
    for token in _tokens(path.read_text(encoding="utf-8-sig")):
        integer_matches = INTEGER_RE.findall(token)
        if integer_matches:
            frame_ids.add(int(integer_matches[-1]))
    return frame_ids


def read_cadica_selected_frame_count(
    frames_dir: Path,
    frames_dirname: str = CADICA_INPUT_FRAMES_DIRNAME,
) -> int:
    selected_frames_path = cadica_selected_frames_path(frames_dir, frames_dirname)
    if not selected_frames_path.is_file():
        raise FileNotFoundError(
            "CADICA selectedFrames file does not exist for "
            f"{frames_dir}: {selected_frames_path}"
        )

    selected_frame_ids = read_selected_frames(selected_frames_path)
    if not selected_frame_ids:
        raise ValueError(f"CADICA selectedFrames file is empty: {selected_frames_path}")
    return len(selected_frame_ids)


def _tokens(text: str) -> list[str]:
    return [token.strip().strip('"\'') for token in TOKEN_SPLIT_RE.split(text) if token.strip()]
