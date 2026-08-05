"""Smoke: prove_it import path + champion load (does not re-score 90 days)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_prove_it_script_imports_and_loads_champion():
    """Drive real script bootstrap: import stack used by prove_it + load_brain."""
    root_s = str(ROOT).replace("\\", "\\\\")
    code = (
        "import os, sys\n"
        f"ROOT = r'{root_s}'\n"
        "for _p in (ROOT, os.path.join(ROOT, 'code')):\n"
        "    if _p not in sys.path:\n"
        "        sys.path.insert(0, _p)\n"
        "from inference.loader import load_brain\n"
        "from core.configs import path as rpath\n"
        "brain, meta = load_brain('PROVEN_SPRINT_row04_clear24_2026-07-20')\n"
        "assert brain is not None, 'champion failed to load'\n"
        "pt = rpath('models', 'PROVEN_SPRINT_row04_clear24_2026-07-20.pt')\n"
        "assert os.path.isfile(pt), pt\n"
        "print('OK load', type(brain).__name__)\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT};{ROOT / 'code'}"
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stdout + "\n" + r.stderr
    assert "OK load" in r.stdout


def test_prove_it_file_has_code_path():
    text = (ROOT / "scripts" / "prove_it.py").read_text(encoding="utf-8")
    assert 'os.path.join(ROOT, "code")' in text or "code" in text
    assert "evaluate" in text
    assert "TGT" in text and "RISK" in text


if __name__ == "__main__":
    test_prove_it_file_has_code_path()
    test_prove_it_script_imports_and_loads_champion()
    print("test_prove_it_launch OK")
