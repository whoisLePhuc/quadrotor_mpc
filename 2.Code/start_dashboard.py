#!/usr/bin/env python3
"""Launch the Streamlit workbench with the active Python environment."""

from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path


def main() -> int:
    if find_spec("streamlit") is None:
        raise SystemExit(
            "Streamlit is not installed. Run `python -m pip install -r requirements-ui.txt`."
        )
    root = Path(__file__).resolve().parent
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(root / "dashboard/Home.py"),
    ]
    return subprocess.call(command, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
