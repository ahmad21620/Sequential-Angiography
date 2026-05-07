from __future__ import annotations

import sys

from _wrapper_utils import get_option_value, has_help_flag, require_existing_path, run_existing_script


def main(argv: list[str] | None = None) -> None:
    forwarded_args = list(sys.argv[1:] if argv is None else argv)
    if not has_help_flag(forwarded_args):
        sweep_root = get_option_value(forwarded_args, "--sweep-root")
        case_root_tree = get_option_value(forwarded_args, "--case-root-tree")
        temporal_results_root = get_option_value(forwarded_args, "--temporal-results-root")

        if sweep_root is not None:
            require_existing_path(sweep_root, "--sweep-root", kind="dir")
        if case_root_tree is not None:
            require_existing_path(case_root_tree, "--case-root-tree", kind="dir")
        if temporal_results_root is not None:
            require_existing_path(temporal_results_root, "--temporal-results-root", kind="dir")

    run_existing_script("stenosis-detection/run_multiview_sweep.py", forwarded_args)


if __name__ == "__main__":
    main()
