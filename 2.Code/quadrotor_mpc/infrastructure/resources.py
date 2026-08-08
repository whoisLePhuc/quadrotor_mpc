"""Resolve bundled inputs and user-owned outputs in source and wheel installs."""

from __future__ import annotations

import sys
from pathlib import Path

_DISTRIBUTION_DATA_DIR = "quadrotor-mpc-sim"


def resource_root() -> Path:
    """Return the directory containing bundled config and model assets."""
    module_path = Path(__file__).resolve()
    for source_root in module_path.parents:
        if (source_root / "pyproject.toml").is_file() and (source_root / "config").is_dir():
            return source_root

    installed_root = Path(sys.prefix) / "share" / _DISTRIBUTION_DATA_DIR
    if (installed_root / "config").is_dir():
        return installed_root

    raise RuntimeError(
        "quadrotor-mpc-sim runtime data is missing; reinstall the package "
        "from a wheel or source checkout"
    )


def resolve_input_path(value: str | Path, *, base: Path | None = None) -> Path:
    """Resolve an input from absolute path, current directory, then package data."""
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    from_cwd = (Path.cwd() / candidate).resolve()
    if from_cwd.exists():
        return from_cwd

    root = resource_root() if base is None else Path(base)
    return (root / candidate).resolve()


def resolve_output_path(value: str | Path) -> Path:
    """Resolve relative outputs against the caller's current directory."""
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (Path.cwd() / candidate).resolve()
