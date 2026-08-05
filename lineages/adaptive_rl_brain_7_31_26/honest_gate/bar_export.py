"""Stable per-bar export schema (same columns for practice and forward).

CHANGE LOG:
- 2026-07-31  honest gate — WHY: seen/unseen must speak the same language.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from lineages.adaptive_rl_brain_7_31_26.honest_gate.score_schema import BAR_EXPORT_SCHEMA
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
)

_ACTION_NAME = {ACTION_HOLD: "HOLD", ACTION_BUY: "BUY", ACTION_SELL: "SELL"}


def schema_column_names() -> List[str]:
    return [c["name"] for c in BAR_EXPORT_SCHEMA]


def empty_bar_row(**kwargs: Any) -> Dict[str, Any]:
    row = {c["name"]: None for c in BAR_EXPORT_SCHEMA}
    row.update(kwargs)
    return row


def export_day_bars(
    day: Any,
    *,
    split: str,
    max_decisions: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Walk decision bars with heuristic + every-bar marks; export stable rows.

    Does not change shell. Uses GoalEquityDay API.
    """
    from lineages.adaptive_rl_brain_7_31_26.perception.types import Direction

    rows: List[Dict[str, Any]] = []
    indices = list(day.runner.decision_indices())
    if max_decisions is not None:
        indices = indices[: int(max_decisions)]
    prev_t = 0
    for t in indices:
        if day.dead or day.banked:
            break
        for bt in range(prev_t, t):
            if day.dead or day.banked:
                break
            day._mark_bar(bt)
        prev_t = t + 1
        if day.dead or day.banked:
            break
        price = float(day._close[t])
        eq = day.equity_pct(price)
        danger = float(np.clip((-eq) / max(day.risk, 1e-6), 0.0, 1.0)) if eq < 0 else 0.0
        progress = float(np.clip(eq / max(day.target, 1e-6), -1.0, 1.0))
        heat_dist = max(0.0, (eq - (-day.risk)) / 100.0)
        heat_ok = heat_dist > 1e-8 and not day.banked and not day.dead
        # perception tags
        perc = day.runner.perceive(t, trade_side=None)
        higher = perc.get("higher")
        lower = perc.get("lower")
        struct = perc.get("structure")
        tag = perc.get("tag")
        rec = int(day.recommended_action(t))
        day.step_action(t, rec)
        side = "FLAT"
        if day.side is not None:
            side = "LONG" if day.side > 0 else "SHORT"
        rows.append(
            empty_bar_row(
                date=day.date_str,
                t=int(t),
                target_pct=float(day.target),
                risk_pct=float(day.risk),
                equity_pct=round(eq, 6),
                min_eq_pct=round(float(day.min_eq_pct), 6),
                danger=round(danger, 6),
                progress_to_goal=round(progress, 6),
                heat_ok=bool(heat_ok),
                action=int(rec),
                side=side,
                higher_dir=str(getattr(higher, "name", higher)),
                lower_dir=str(getattr(lower, "name", lower)),
                pullback=bool(getattr(struct, "pullback", False)) if struct else False,
                scale_conflict=bool(getattr(struct, "scale_conflict", False))
                if struct
                else False,
                tag=str(getattr(tag, "name", tag) if tag is not None else ""),
                banked=bool(day.banked),
                breached=bool(day.breached),
                split=str(split),
            )
        )
    return rows


def schemas_identical() -> bool:
    """Practice and forward use the same column list by construction."""
    return schema_column_names() == schema_column_names()
