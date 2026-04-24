from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class KeyframeCandidate:
    frame_index: int
    name: str
    source_path: Path
    score: float


@dataclass(slots=True, frozen=True)
class ExtractionResult:
    frames_dir: Path
    output_dir: Path
    selected_count: int
    skipped: bool = False
