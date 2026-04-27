from __future__ import annotations

import sys

from _wrapper_utils import get_option_value, has_help_flag, require_existing_path, run_existing_script


def main(argv: list[str] | None = None) -> None:
    forwarded_args = list(sys.argv[1:] if argv is None else argv)
    if not has_help_flag(forwarded_args):
        results_root = get_option_value(forwarded_args, "--results-root")
        weak_labels = get_option_value(forwarded_args, "--weak-labels")

        if results_root is not None:
            require_existing_path(results_root, "--results-root", kind="dir")
        if weak_labels is not None:
            require_existing_path(weak_labels, "--weak-labels", kind="file")

    run_existing_script("stenosis-detection/run_benchmark.py", forwarded_args)


if __name__ == "__main__":
    main()
