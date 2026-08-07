"""CLI entry — GSD ship for forward principle learning.

Usage (repo root):
  $env:PYTHONPATH = ".;code"
  python lineages/adaptive_rl_brain_7_31_26/forward_principle_learn/run_forward_learn_cycle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lineages.adaptive_rl_brain_7_31_26.forward_principle_learn.train_principle_student import (  # noqa: E402
    main,
)

if __name__ == "__main__":
    main()
