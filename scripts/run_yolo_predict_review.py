from __future__ import annotations

import sys
from pathlib import Path

from _wrapper_utils import get_option_value, has_help_flag, require_existing_path, resolve_repo_path, run_existing_script


def main(argv: list[str] | None = None) -> None:
    forwarded_args = list(sys.argv[1:] if argv is None else argv)
    if not has_help_flag(forwarded_args):
        weights = get_option_value(forwarded_args, "--weights")
        source = get_option_value(forwarded_args, "--source")
        if weights is not None:
            require_existing_path(weights, "--weights", kind="file")
        if source is not None and _looks_like_local_path(source):
            require_existing_path(source, "--source")

    run_existing_script("stenosis-detection/run_yolo_predict_review.py", forwarded_args)


def _looks_like_local_path(value: str) -> bool:
    if "://" in value:
        return False
    if value.isdigit():
        return False
    return resolve_repo_path(value).exists() or Path(value).suffix.lower() != ""


if __name__ == "__main__":
    main()
