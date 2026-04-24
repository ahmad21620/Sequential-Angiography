from __future__ import annotations

import sys

from _wrapper_utils import (
    has_help_flag,
    get_option_value,
    require_existing_path,
    run_existing_script,
)


DEFAULT_CHECKPOINT = "vessel-segmentation/checkpoints/best_model.pt"


def main(argv: list[str] | None = None) -> None:
    forwarded_args = list(sys.argv[1:] if argv is None else argv)
    if not has_help_flag(forwarded_args):
        input_path = get_option_value(forwarded_args, "--input")
        if input_path is not None:
            require_existing_path(input_path, "--input")
            checkpoint_path = get_option_value(forwarded_args, "--checkpoint") or DEFAULT_CHECKPOINT
            require_existing_path(checkpoint_path, "--checkpoint", kind="file")

    run_existing_script("vessel-segmentation/segment_retinal_images.py", forwarded_args)


if __name__ == "__main__":
    main()
