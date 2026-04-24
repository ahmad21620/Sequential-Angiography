from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def has_help_flag(argv: Sequence[str]) -> bool:
    return any(argument in {"-h", "--help"} for argument in argv)


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def require_existing_path(value: str, label: str, *, kind: str = "path") -> None:
    path = resolve_repo_path(value)
    if not path.exists():
        _exit_with_error(f"{label} does not exist: {path}")
    if kind == "file" and not path.is_file():
        _exit_with_error(f"{label} is not a file: {path}")
    if kind == "dir" and not path.is_dir():
        _exit_with_error(f"{label} is not a directory: {path}")


def get_option_value(argv: Sequence[str], option: str) -> str | None:
    prefix = f"{option}="
    for index, argument in enumerate(argv):
        if argument.startswith(prefix):
            return argument[len(prefix) :]
        if argument == option and index + 1 < len(argv):
            return argv[index + 1]
    return None


def get_option_values(argv: Sequence[str], option: str) -> list[str]:
    values: list[str] = []
    index = 0
    prefix = f"{option}="
    while index < len(argv):
        argument = argv[index]
        if argument.startswith(prefix):
            values.append(argument[len(prefix) :])
            index += 1
            continue
        if argument == option:
            index += 1
            while index < len(argv) and not argv[index].startswith("-"):
                values.append(argv[index])
                index += 1
            continue
        index += 1
    return values


def first_positional_argument(argv: Sequence[str], *, value_options: set[str], flag_options: set[str]) -> str | None:
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--":
            return argv[index + 1] if index + 1 < len(argv) else None
        if argument.startswith("-"):
            option = argument.split("=", 1)[0]
            if option in flag_options or "=" in argument:
                index += 1
                continue
            if option in value_options or index + 1 < len(argv):
                index += 2
                continue
        return argument
    return None


def run_existing_script(relative_script_path: str, argv: Sequence[str]) -> None:
    script_path = REPO_ROOT / relative_script_path
    if not script_path.is_file():
        _exit_with_error(f"wrapper target script is missing: {script_path}")
    completed = subprocess.run(
        [sys.executable, str(script_path), *argv],
        cwd=REPO_ROOT,
        check=False,
    )
    raise SystemExit(completed.returncode)


def _exit_with_error(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(2)
