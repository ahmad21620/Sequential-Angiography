from __future__ import annotations

from stenosis_detection.cli import (
    _validate_batch_mode,
    _validate_single_mode,
    build_parser,
    main,
)

__all__ = [
    "build_parser",
    "main",
    "_validate_batch_mode",
    "_validate_single_mode",
]


if __name__ == "__main__":
    raise SystemExit(main())
