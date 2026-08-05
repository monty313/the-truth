"""Official Sets + Sub-Sets for adaptive_rl_brain_7_31_26.

CHANGE LOG:
- 2026-08-04  MARK SETS LAW pin + assert — WHY: Monty lock Mark-on-chart:
  LTF=first (pullback/cont/add); HTF=last two (trend confirm); scan all 4.
  See MARK_SETS_LAW.md. Do not edit stacks without rewriting that law + tests.
- 2026-07-31  created — WHY: Phase 1 data structures from SPEC_PHASE1
  (4 Official Sets, 5 Sub-Sets; roles relative Entry vs Confirmation).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from lineages.adaptive_rl_brain_7_31_26.perception.types import OfficialSet, SubSet


# ---- MARK SETS LAW (immutable without human rewrite of MARK_SETS_LAW.md) ----
# LTF (first) = pullbacks / continuations / adds.
# HTF (second, third) = trend confirmation (two TFs).
MARK_SETS_LAW: Tuple[Tuple[int, str, str, Tuple[str, str]], ...] = (
    (1, "micro", "1m", ("15m", "30m")),
    (2, "intraday", "5m", ("30m", "1h")),
    (3, "swing", "15m", ("1h", "4h")),
    (4, "macro", "30m", ("4h", "1d")),
)

# ---- Official Sets (full strength) — MUST match MARK_SETS_LAW ----
# Entry = first; Confirmation = remaining two higher TFs.
OFFICIAL_SETS: Tuple[OfficialSet, ...] = tuple(
    OfficialSet(sid, name, ltf, htfs) for sid, name, ltf, htfs in MARK_SETS_LAW
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


def mark_sets_law_table() -> List[dict]:
    """Human/agent table: LTF first, HTF last two — Mark on the chart."""
    return [
        {
            "set_id": s.set_id,
            "name": s.name,
            "ltf_entry": s.entry_tf,
            "htf_confirm": list(s.confirmation_tfs),
            "stack": list(s.tfs),
            "ltf_job": "pullback_continuation_add",
            "htf_job": "trend_confirm",
        }
        for s in OFFICIAL_SETS
    ]


def assert_mark_sets_law(
    stacks: Sequence[Tuple[str, str, str]] | None = None,
) -> None:
    """Hard fail if official stacks drift from MARK SETS LAW.

    Expected stacks (LTF, HTF1, HTF2):
      1m,15m,30m · 5m,30m,1h · 15m,1h,4h · 30m,4h,1d
    """
    expected = [
        ("1m", "15m", "30m"),
        ("5m", "30m", "1h"),
        ("15m", "1h", "4h"),
        ("30m", "4h", "1d"),
    ]
    if stacks is None:
        stacks = [s.tfs for s in OFFICIAL_SETS]
    got = [tuple(x) for x in stacks]
    if len(got) != 4:
        raise AssertionError(f"MARK SETS LAW requires 4 sets, got {len(got)}")
    for i, (e, g) in enumerate(zip(expected, got), start=1):
        if e != g:
            raise AssertionError(
                f"MARK SETS LAW broken on set {i}: expected {e}, got {g}. "
                "Rewrite MARK_SETS_LAW.md + tests only with Monty order."
            )
    # Roles: first is LTF, last two HTF
    for s in OFFICIAL_SETS:
        if s.entry_tf != s.tfs[0]:
            raise AssertionError(f"set {s.set_id}: LTF must be first TF")
        if s.confirmation_tfs != s.tfs[1:]:
            raise AssertionError(f"set {s.set_id}: HTF confirm must be last two TFs")


# Fail import if law is ever hand-edited wrong
assert_mark_sets_law()
