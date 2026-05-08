from __future__ import annotations

from stenosis_detection.yolo.predict_review import build_parser, main, run_prediction_review

__all__ = ["build_parser", "main", "run_prediction_review"]


if __name__ == "__main__":
    raise SystemExit(main())
