#!/usr/bin/env python3
"""Launch the Streamlit workbench with the active Python environment."""

from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

from quadrotor_mpc.interfaces import dashboard


def main() -> int:
    if find_spec("streamlit") is None:
        raise SystemExit(
            "Streamlit is not installed. Run `python -m pip install -r requirements-ui.txt`."
        )
    dashboard_root = Path(dashboard.__path__[0])
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_root / "Home.py"),
    ]
    return subprocess.call(command, cwd=dashboard_root)


if __name__ == "__main__":
    raise SystemExit(main())
