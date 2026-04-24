from __future__ import annotations

import os
from pathlib import Path

from angio_keyframes.images import is_supported_image_name, list_image_files


def discover_frame_directories(root: Path, frames_dirname: str = "frames") -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    if list_image_files(root):
        return [root]

    discovered = {
        Path(directory_path)
        for directory_path, _directory_names, file_names in os.walk(root)
        if any(is_supported_image_name(file_name) for file_name in file_names)
    }

    return sorted(discovered)
