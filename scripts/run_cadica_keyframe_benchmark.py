from __future__ import annotations

import sys

from _wrapper_utils import get_option_value, has_help_flag, require_existing_path, run_existing_script


def main(argv: list[str] | None = None) -> None:
    forwarded_args = list(sys.argv[1:] if argv is None else argv)
    if not has_help_flag(forwarded_args):
        cadica_root = get_option_value(forwarded_args, "--cadica-root")
        extracted_keyframes_root = get_option_value(forwarded_args, "--extracted-keyframes-root")

        if cadica_root is not None:
            require_existing_path(cadica_root, "--cadica-root", kind="dir")
        if extracted_keyframes_root is not None:
            require_existing_path(extracted_keyframes_root, "--extracted-keyframes-root", kind="dir")

    run_existing_script("keyframes-extraction/run_cadica_keyframe_benchmark.py", forwarded_args)


if __name__ == "__main__":
    main()
