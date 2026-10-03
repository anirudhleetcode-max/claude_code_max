"""Generated documentation must match the code."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_generated_reference_docs_are_current() -> None:
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_docs.py"), "--check"], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
