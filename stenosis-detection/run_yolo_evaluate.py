from __future__ import annotations

from stenosis_detection.yolo.evaluate import build_parser, main, run_evaluation

__all__ = ["build_parser", "main", "run_evaluation"]


if __name__ == "__main__":
    raise SystemExit(main())
