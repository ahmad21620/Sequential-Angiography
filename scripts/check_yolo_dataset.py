from __future__ import annotations

import sys

from _wrapper_utils import get_option_value, has_help_flag, require_existing_path, run_existing_script


def main(argv: list[str] | None = None) -> None:
    forwarded_args = list(sys.argv[1:] if argv is None else argv)
    if not has_help_flag(forwarded_args):
        data = get_option_value(forwarded_args, "--data")
        if data is not None:
            require_existing_path(data, "--data", kind="file")

    run_existing_script("stenosis-detection/check_yolo_dataset.py", forwarded_args)


if __name__ == "__main__":
    main()
