from __future__ import annotations

import sys
from pathlib import Path

from _wrapper_utils import (
    get_option_value,
    has_help_flag,
    require_existing_path,
    resolve_repo_path,
    run_existing_script,
)


def main(argv: list[str] | None = None) -> None:
    forwarded_args = list(sys.argv[1:] if argv is None else argv)
    if not has_help_flag(forwarded_args):
        case_root = get_option_value(forwarded_args, "--case-root")
        input_json = get_option_value(forwarded_args, "--input-json")

        if case_root is not None:
            require_existing_path(case_root, "--case-root", kind="dir")
            views_json = resolve_repo_path(str(Path(case_root) / "views.json"))
            if not views_json.is_file():
                print(f"Error: --case-root does not contain views.json: {views_json}", file=sys.stderr)
                raise SystemExit(2)
        if input_json is not None:
            require_existing_path(input_json, "--input-json", kind="file")

    run_existing_script("stenosis-detection/run_multiview_fusion.py", forwarded_args)


if __name__ == "__main__":
    main()
