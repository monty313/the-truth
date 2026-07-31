"""Official Sets + Sub-Sets for adaptive_rl_brain_7_31_26.

CHANGE LOG:
- 2026-07-31  created — WHY: Phase 1 data structures from SPEC_PHASE1
  (4 Official Sets, 5 Sub-Sets; roles relative Entry vs Confirmation).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from lineages.adaptive_rl_brain_7_31_26.perception.types import OfficialSet, SubSet


# ---- Official Sets (full strength) ----
# Entry = first; Confirmation = remaining two higher TFs.
OFFICIAL_SETS: Tuple[OfficialSet, ...] = (
    OfficialSet(1, "micro", "1m", ("15m", "30m")),
    OfficialSet(2, "intraday", "5m", ("30m", "1h")),
    OfficialSet(3, "swing", "15m", ("1h", "4h")),
    OfficialSet(4, "macro", "30m", ("4h", "1d")),
)

# ---- Sub-Sets (weaker / lower confidence) ----
# first TF = Entry, second = Confirmation
SUB_SETS: Tuple[SubSet, ...] = (
    SubSet("A", "1m", "5m"),
    SubSet("B", "5m", "15m"),
    SubSet("C", "15m", "30m"),
    SubSet("D", "1h", "4h"),
    SubSet("E", "4h", "1d"),
)

_OFFICIAL_BY_ID: Dict[int, OfficialSet] = {s.set_id: s for s in OFFICIAL_SETS}
_SUB_BY_ID: Dict[str, SubSet] = {s.sub_id: s for s in SUB_SETS}


def get_official(set_id: int) -> OfficialSet:
    try:
        return _OFFICIAL_BY_ID[set_id]
    except KeyError as e:
        raise KeyError(f"unknown Official Set id {set_id!r}; expected 1..4") from e


def get_sub(sub_id: str) -> SubSet:
    key = sub_id.strip().upper()
    try:
        return _SUB_BY_ID[key]
    except KeyError as e:
        raise KeyError(f"unknown Sub-Set id {sub_id!r}; expected A..E") from e


def all_official() -> List[OfficialSet]:
    return list(OFFICIAL_SETS)


def all_subs() -> List[SubSet]:
    return list(SUB_SETS)


def entry_tf(set_or_sub: OfficialSet | SubSet) -> str:
    """Relative role: first TF is always Entry."""
    return set_or_sub.entry_tf


def confirmation_tfs(set_or_sub: OfficialSet | SubSet) -> Tuple[str, ...]:
    """Relative role: remaining TFs are Confirmation."""
    if isinstance(set_or_sub, OfficialSet):
        return set_or_sub.confirmation_tfs
    return (set_or_sub.confirmation_tf,)
