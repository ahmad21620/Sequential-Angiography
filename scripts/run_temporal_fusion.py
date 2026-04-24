from __future__ import annotations

import sys

from _wrapper_utils import (
    get_option_value,
    get_option_values,
    has_help_flag,
    require_existing_path,
    run_existing_script,
)


def main(argv: list[str] | None = None) -> None:
    forwarded_args = list(sys.argv[1:] if argv is None else argv)
    if not has_help_flag(forwarded_args):
        results_root = get_option_value(forwarded_args, "--results-root")
        if results_root is not None:
            require_existing_path(results_root, "--results-root", kind="dir")

        for frame_result in get_option_values(forwarded_args, "--frame-results"):
            require_existing_path(frame_result, "--frame-results", kind="file")

    run_existing_script("stenosis-detection/run_temporal_fusion.py", forwarded_args)


if __name__ == "__main__":
    main()
