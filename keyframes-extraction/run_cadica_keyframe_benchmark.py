from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from angio_keyframes.cadica_benchmark import build_parser, main, run_cadica_keyframe_benchmark

__all__ = ["build_parser", "main", "run_cadica_keyframe_benchmark"]


if __name__ == "__main__":
    raise SystemExit(main())
