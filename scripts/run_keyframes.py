from __future__ import annotations

import sys

from _wrapper_utils import (
    first_positional_argument,
    get_option_value,
    has_help_flag,
    require_existing_path,
    run_existing_script,
)


VALUE_OPTIONS = {
    "--limit",
    "--baseline-frames",
    "--smoothing-window",
    "--window-mode",
    "--workers",
    "--backend",
    "--output-root",
    "--frames-dirname",
    "--input-root",
}
FLAG_OPTIONS = {
    "--overwrite",
    "--skip-existing",
    "--cadica-selected-frame-counts",
}


def main(argv: list[str] | None = None) -> None:
    forwarded_args = list(sys.argv[1:] if argv is None else argv)
    if not has_help_flag(forwarded_args):
        input_path = get_option_value(forwarded_args, "--input-root")
        if input_path is None:
            input_path = first_positional_argument(
                forwarded_args,
                value_options=VALUE_OPTIONS,
                flag_options=FLAG_OPTIONS,
            )
        if input_path is not None:
            require_existing_path(input_path, "input_path")

    run_existing_script("keyframes-extraction/keyframes_extraction.py", forwarded_args)


if __name__ == "__main__":
    main()
