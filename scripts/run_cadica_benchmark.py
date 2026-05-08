from __future__ import annotations

import sys

from _wrapper_utils import get_option_value, has_help_flag, require_existing_path, run_existing_script


def main(argv: list[str] | None = None) -> None:
    forwarded_args = list(sys.argv[1:] if argv is None else argv)
    if not has_help_flag(forwarded_args):
        manifest = get_option_value(forwarded_args, "--manifest")
        frame_results_root = get_option_value(forwarded_args, "--frame-results-root")
        temporal_results_root = get_option_value(forwarded_args, "--temporal-results-root")
        multiview_results_root = get_option_value(forwarded_args, "--multiview-results-root")

        if manifest is not None:
            require_existing_path(manifest, "--manifest", kind="file")
        if frame_results_root is not None:
            require_existing_path(frame_results_root, "--frame-results-root", kind="dir")
        if temporal_results_root is not None:
            require_existing_path(temporal_results_root, "--temporal-results-root", kind="dir")
        if multiview_results_root is not None:
            require_existing_path(multiview_results_root, "--multiview-results-root", kind="dir")

    run_existing_script("stenosis-detection/run_cadica_benchmark.py", forwarded_args)


if __name__ == "__main__":
    main()
