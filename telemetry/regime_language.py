"""Regime language — name Gravity observations so strategy can emerge and be audited.

5W+I -----------------------------------------------------------------
WHO:   Fable 5 for Monty.
WHAT:  Maps existing observation flags (cont/pull/rev/masks) into the shared
       regime vocabulary (HTF role, LTF setup, skip reasons). Documents state
       on every decision without changing the observation space.
WHEN:  2026-07-24.
WHY:   Bot needs language to form strategy combinations; Monty needs a record
       of why HTF trend did not produce LTF pullback/continuation entries.
INTERCONNECTED WITH: doctrine/REGIME_LANGUAGE.md, telemetry/mind_probe.py,
       features/engine.py (cont/pull/rev), doctrine/STANDING_LAWS.md.
----------------------------------------------------------------------

CHANGE LOG:
- 2026-07-24  created — WHY: regime language + mandatory decision documentation.
# NEXT EDITOR: append dated WHY; keep this line.
"""
from __future__ import annotations

from typing import Any


HTF_REGIMES = (
    "trend_bull",
    "trend_bear",
    "range",
    "transition",
    "unknown",
)
LTF_SETUPS = (
    "pullback",
    "continuation",
    "reversal",
    "none",
)
SKIP_REASONS = (
    "policy_hold",
    "mask_veto",
    "no_ltf_setup",
    "no_htf_bias",
    "flat_ok",
    "acted",
)

TARGET_SETS = {
    "A": {"ltf": "1m", "htfs": ["15m", "30m"]},
    "B": {"ltf": "5m", "htfs": ["1h", "4h"]},
    "C": {"ltf": "15m", "htfs": ["4h", "1d"]},
}

CODE_SETS = {
    "set1": {"ltf": "1m", "htfs": ["15m", "30m"]},
    "set2": {"ltf": "5m", "htfs": ["30m", "1h"]},
    "set3": {"ltf": "15m", "htfs": ["1h", "4h"]},
    "set4": {"ltf": "30m", "htfs": ["4h", "1d"]},
}


def classify_htf_regime(
    cont_buy: bool,
    cont_sell: bool,
    rev_buy: bool,
    rev_sell: bool,
    pull_buy: bool = False,
    pull_sell: bool = False,
) -> str:
    if rev_buy or rev_sell:
        return "transition"
    if cont_buy and not cont_sell:
        return "trend_bull"
    if cont_sell and not cont_buy:
        return "trend_bear"
    if pull_buy and not pull_sell and not cont_sell:
        return "trend_bull"
    if pull_sell and not pull_buy and not cont_buy:
        return "trend_bear"
    if not cont_buy and not cont_sell and not pull_buy and not pull_sell:
        return "range"
    return "unknown"


def classify_ltf_setup(
    cont_buy: bool,
    cont_sell: bool,
    pull_buy: bool,
    pull_sell: bool,
    rev_buy: bool,
    rev_sell: bool,
) -> tuple[str, str]:
    if pull_buy:
        return "pullback", "buy"
    if pull_sell:
        return "pullback", "sell"
    if cont_buy:
        return "continuation", "buy"
    if cont_sell:
        return "continuation", "sell"
    if rev_buy:
        return "reversal", "buy"
    if rev_sell:
        return "reversal", "sell"
    return "none", "none"


def _side_from_op(op_name: str) -> str:
    if "long" in op_name:
        return "buy"
    if "short" in op_name:
        return "sell"
    return "none"


def _is_entry_op(op_name: str) -> bool:
    return op_name in {
        "open_long", "open_short", "add_long", "add_short",
        "probe_long", "probe_short",
    }


def document_decision(
    *,
    cont_buy: bool,
    cont_sell: bool,
    pull_buy: bool,
    pull_sell: bool,
    rev_buy: bool,
    rev_sell: bool,
    mask_buy_blocked: bool,
    mask_sell_blocked: bool,
    chosen_op_name: str,
) -> dict[str, Any]:
    htf = classify_htf_regime(cont_buy, cont_sell, rev_buy, rev_sell, pull_buy, pull_sell)
    setup, setup_side = classify_ltf_setup(
        cont_buy, cont_sell, pull_buy, pull_sell, rev_buy, rev_sell
    )
    op_side = _side_from_op(chosen_op_name)
    entered = _is_entry_op(chosen_op_name)
    htf_trending = htf in ("trend_bull", "trend_bear")
    setup_alive = setup in ("pullback", "continuation")

    skip_reason = "acted"
    if entered:
        skip_reason = "acted"
    elif setup_side == "buy" and mask_buy_blocked:
        skip_reason = "mask_veto"
    elif setup_side == "sell" and mask_sell_blocked:
        skip_reason = "mask_veto"
    elif htf_trending and setup_alive and chosen_op_name == "hold":
        skip_reason = "policy_hold"
    elif htf_trending and not setup_alive:
        skip_reason = "no_ltf_setup"
    elif not htf_trending and setup == "none":
        skip_reason = "flat_ok"
    elif not htf_trending:
        skip_reason = "no_htf_bias"
    elif chosen_op_name == "hold":
        skip_reason = "policy_hold"
    else:
        skip_reason = "flat_ok"

    why = _why_text(htf, setup, setup_side, chosen_op_name, skip_reason)
    return {
        "htf_regime": htf,
        "ltf_setup": setup,
        "setup_side": setup_side,
        "mask_buy_blocked": mask_buy_blocked,
        "mask_sell_blocked": mask_sell_blocked,
        "chosen_op": chosen_op_name,
        "entered": entered,
        "matched_setup": entered and op_side == setup_side and setup_side != "none",
        "skip_reason": skip_reason,
        "why": why,
    }


def _why_text(htf: str, setup: str, setup_side: str, op: str, skip: str) -> str:
    if skip == "acted":
        return f"HTF={htf}; LTF={setup}/{setup_side}; policy chose {op}."
    if skip == "policy_hold":
        return (
            f"HTF={htf} (trending bias on) and LTF {setup}/{setup_side} was visible, "
            f"but policy chose {op}. Candidate Policy issue (incentives/fear), not blindness."
        )
    if skip == "mask_veto":
        return f"HTF={htf}; LTF={setup}/{setup_side}; forever-mask blocked that side."
    if skip == "no_ltf_setup":
        return (
            f"HTF={htf} (bias on) but no LTF pullback/continuation/reversal flag — "
            f"no execution trigger under Gravity composition."
        )
    if skip == "no_htf_bias":
        return f"HTF={htf}; no clear directional permission — hold is consistent with Law 1 bias gate."
    return f"HTF={htf}; LTF={setup}/{setup_side}; op={op}; skip={skip}."


def summarize_day_skips(docs: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter
    c = Counter(d.get("skip_reason", "unknown") for d in docs)
    policy_holds = [
        d for d in docs
        if d.get("skip_reason") == "policy_hold" and d.get("ltf_setup") in ("pullback", "continuation")
    ]
    return {
        "n_decisions": len(docs),
        "skip_counts": dict(c),
        "n_policy_hold_on_setup": len(policy_holds),
        "n_mask_veto": c.get("mask_veto", 0),
        "n_no_ltf_setup": c.get("no_ltf_setup", 0),
        "n_acted": c.get("acted", 0),
    }
