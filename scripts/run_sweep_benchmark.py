from __future__ import annotations

import sys

from _wrapper_utils import get_option_value, has_help_flag, require_existing_path, run_existing_script


def main(argv: list[str] | None = None) -> None:
    forwarded_args = list(sys.argv[1:] if argv is None else argv)
    if not has_help_flag(forwarded_args):
        sweep_root = get_option_value(forwarded_args, "--sweep-root")
        weak_labels = get_option_value(forwarded_args, "--weak-labels")

        if sweep_root is not None:
            require_existing_path(sweep_root, "--sweep-root", kind="dir")
        if weak_labels is not None:
            require_existing_path(weak_labels, "--weak-labels", kind="file")

    run_existing_script("stenosis-detection/run_sweep_benchmark.py", forwarded_args)


if __name__ == "__main__":
    main()
