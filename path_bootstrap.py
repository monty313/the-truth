"""Put repo root + code/ on sys.path (FinRL-clean layout).

Import this first in scripts if needed, or set:
  PYTHONPATH=<repo>;<repo>/code
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
