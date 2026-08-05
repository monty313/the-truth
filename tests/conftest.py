"""Ensure repo root + code/ are on sys.path for lineage + production packages.

CRITICAL: ``tests/lineages/`` is a pytest package and would shadow the real
``lineages/`` tree if ``tests`` is earlier on sys.path. Always force repo root
first and drop any shadowed ``lineages`` modules before collection imports.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CODE = _ROOT / "code"
_TESTS = Path(__file__).resolve().parent

# Prefer repo root over tests/ so ``import lineages`` resolves to the real package.
_cleaned = [p for p in sys.path if Path(p).resolve() != _TESTS.resolve()]
sys.path[:] = [str(_ROOT), str(_CODE)] + [
    p for p in _cleaned if p not in (str(_ROOT), str(_CODE), "")
]

# Drop any already-imported shadow from tests/lineages
for _name in list(sys.modules):
    if _name == "lineages" or _name.startswith("lineages."):
        mod = sys.modules.get(_name)
        f = getattr(mod, "__file__", "") or ""
        if "tests" in f.replace("\\", "/") and "/tests/lineages" in f.replace("\\", "/"):
            del sys.modules[_name]
        elif f and "tests\\lineages" in f:
            del sys.modules[_name]
