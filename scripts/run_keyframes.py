from __future__ import annotations

import sys

from _wrapper_utils import (
    first_positional_argument,
    has_help_flag,
    require_existing_path,
    run_existing_script,
)


VALUE_OPTIONS = {
    "--limit",
    "--baseline-frames",
    "--smoothing-window",
    "--workers",
    "--backend",
    "--output-root",
    "--frames-dirname",
}
FLAG_OPTIONS = {
    "--overwrite",
    "--skip-existing",
}


def main(argv: list[str] | None = None) -> None:
    forwarded_args = list(sys.argv[1:] if argv is None else argv)
    if not has_help_flag(forwarded_args):
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
