from __future__ import annotations

from stenosis_detection.yolo.train import build_parser, main, run_training

__all__ = ["build_parser", "main", "run_training"]


if __name__ == "__main__":
    raise SystemExit(main())
