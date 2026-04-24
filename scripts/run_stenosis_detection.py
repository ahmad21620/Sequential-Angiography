from __future__ import annotations

import sys

from _wrapper_utils import (
    get_option_value,
    has_help_flag,
    require_existing_path,
    run_existing_script,
)


def main(argv: list[str] | None = None) -> None:
    forwarded_args = list(sys.argv[1:] if argv is None else argv)
    if not has_help_flag(forwarded_args):
        image_path = get_option_value(forwarded_args, "--image")
        mask_path = get_option_value(forwarded_args, "--mask")
        images_root = get_option_value(forwarded_args, "--images-root")
        masks_root = get_option_value(forwarded_args, "--masks-root")

        if image_path is not None:
            require_existing_path(image_path, "--image", kind="file")
        if mask_path is not None:
            require_existing_path(mask_path, "--mask", kind="file")
        if images_root is not None:
            require_existing_path(images_root, "--images-root", kind="dir")
        if masks_root is not None:
            require_existing_path(masks_root, "--masks-root", kind="dir")

    run_existing_script("stenosis-detection/run_stenosis_detection.py", forwarded_args)


if __name__ == "__main__":
    main()
