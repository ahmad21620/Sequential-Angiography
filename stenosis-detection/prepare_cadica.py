from __future__ import annotations

from stenosis_detection.cadica.prepare import build_parser, main, prepare_cadica_for_pipeline

__all__ = ["build_parser", "main", "prepare_cadica_for_pipeline"]


if __name__ == "__main__":
    raise SystemExit(main())
