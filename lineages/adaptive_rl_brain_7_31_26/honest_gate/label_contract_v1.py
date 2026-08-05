"""Label Contract V1 — pure label functions + frozen field schema.

CHANGE LOG:
- 2026-07-31  Principle 9 logger foundation — WHY: factual decision explanations
  before any attention/training. Shell/heuristic unchanged.

Source of truth for meanings: 00_LABEL_CONTRACT_V1.md
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from lineages.adaptive_rl_brain_7_31_26.perception.types import Direction, VelocityStrength
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
)

# Locked to equity_day.POINT_SIZE (do not import equity_day here — keeps pure labels light).
POINT_SIZE = 0.01

LABEL_CONTRACT_VERSION = "label_contract_v1"

# Decision-bar required fields (order is stable for schema equality tests).
DECISION_FIELD_NAMES: Tuple[str, ...] = (
    "label_contract_version",
    "meaning_hash",
    "date",
    "t",
    "split",
    "target_pct",
    "risk_pct",
    "htf_trend_dir",
    "htf_trend_strength",
    "ltf_trend_dir",
    "alignment",
    "channel_position",
    "channel_slope",
    "cci_state",
    "rsi_state",
    "momentum_velocity",
    "momentum_velocity_change",
    "momentum_vs_direction",
    "market_condition",
    "pullback_state",
    "scale_conflict",
    "agreement_profile",
    "session_phase",
    "equity_pct",
    "target_remaining_pct",
    "risk_remaining_pct",
    "heat_ok",
    "spread_condition",
    "remaining_opportunity_est",
    "action",
    "position_before",
    "entry_number",
    "reversal_number",
    "reversed_while_htf_unchanged",
    "entry_reason",
    "exit_reason",
    "no_trade_reason",
    "future_bar_used",
)

# End-of-day required fields.
EOD_FIELD_NAMES: Tuple[str, ...] = (
    "label_contract_version",
    "meaning_hash",
    "date",
    "split",
    "target_pct",
    "risk_pct",
    "cleared",
    "breached",
    "banked",
    "pnl_pct",
    "min_eq_pct",
    "dist_to_target_pct",
    "n_entries",
    "reversal_count",
    "day_activity_class",
    "n_decision_rows",
)

ACTION_NAME = {ACTION_HOLD: "HOLD", ACTION_BUY: "BUY", ACTION_SELL: "SELL"}

_VEL_RANK = {
    "none": 0,
    "weak": 1,
    "medium": 2,
    "strong": 3,
}


def dir_to_label(d: Any) -> str:
    """Map Direction enum / int / name → bullish|bearish|neutral|unknown."""
    if d is None:
        return "unknown"
    if isinstance(d, Direction):
        if d == Direction.BULL:
            return "bullish"
        if d == Direction.BEAR:
            return "bearish"
        if d == Direction.NEUTRAL:
            return "neutral"
        return "unknown"
    if isinstance(d, (int, float)):
        if int(d) == 1:
            return "bullish"
        if int(d) == -1:
            return "bearish"
        if int(d) == 0:
            return "neutral"
        return "unknown"
    name = str(getattr(d, "name", d)).upper()
    if name in ("BULL", "BULLISH", "1"):
        return "bullish"
    if name in ("BEAR", "BEARISH", "-1"):
        return "bearish"
    if name in ("NEUTRAL", "FLAT", "0"):
        return "neutral"
    return "unknown"


def velocity_to_label(v: Any) -> str:
    if v is None:
        return "unknown"
    if isinstance(v, VelocityStrength):
        return str(v.value)
    s = str(getattr(v, "value", v)).lower()
    if s in ("strong", "medium", "weak", "none"):
        return s
    return "unknown"


def alignment_label(htf: str, ltf: str) -> str:
    if htf == "unknown" or ltf == "unknown":
        return "unknown"
    if htf == "neutral" or ltf == "neutral":
        return "neutral"
    if htf == ltf:
        return "aligned"
    return "conflicting"


def channel_position_label(
    *,
    both_above: bool,
    both_below: bool,
    inside_both: bool,
    ready: bool,
) -> str:
    if not ready:
        return "unknown"
    if both_above:
        return "above"
    if both_below:
        return "below"
    if inside_both:
        return "inside"
    return "unknown"


def channel_slope_label(mid_t: float, mid_prev: float) -> str:
    if not (math.isfinite(mid_t) and math.isfinite(mid_prev)):
        return "unknown"
    delta = mid_t - mid_prev
    eps = 0.5 * POINT_SIZE
    if abs(delta) <= eps:
        return "flat"
    if delta > eps:
        return "rising"
    return "falling"


def cci_state_label(
    cci30: float,
    cci100: float,
    cci30_prev: Optional[float],
) -> str:
    if not (math.isfinite(cci30) and math.isfinite(cci100)):
        return "unknown"
    if cci30 >= 100 and cci100 >= 100:
        return "extended_high"
    if cci30 <= -100 and cci100 <= -100:
        return "extended_low"
    if cci30_prev is not None and math.isfinite(cci30_prev):
        if cci30 > cci30_prev:
            return "strengthening"
        if cci30 < cci30_prev:
            return "weakening"
    if cci30 > 0 and cci100 > 0:
        return "above_zero"
    if cci30 < 0 and cci100 < 0:
        return "below_zero"
    return "mixed_zero"


def rsi_state_label(
    rsi5: float,
    rsi14: float,
    rsi14_prev: Optional[float],
) -> str:
    if not (math.isfinite(rsi14) and math.isfinite(rsi5)):
        return "unknown"
    if rsi14 >= 70:
        return "extended_high"
    if rsi14 <= 30:
        return "extended_low"
    if rsi14_prev is not None and math.isfinite(rsi14_prev):
        if rsi14 > rsi14_prev:
            return "strengthening"
        if rsi14 < rsi14_prev:
            return "weakening"
    if rsi14 > 50 and rsi5 > 50:
        return "above_50"
    if rsi14 < 50 and rsi5 < 50:
        return "below_50"
    return "mixed_50"


def momentum_velocity_change_label(level_t: str, level_prev: str) -> str:
    if level_t == "unknown" or level_prev == "unknown":
        return "unknown"
    if level_t not in _VEL_RANK or level_prev not in _VEL_RANK:
        return "unknown"
    rt, rp = _VEL_RANK[level_t], _VEL_RANK[level_prev]
    if rt > rp:
        return "strengthening"
    if rt < rp:
        return "weakening"
    return "flat"


def momentum_vs_direction_label(htf: str, osc_dir: str) -> str:
    if htf == "unknown" or osc_dir == "unknown":
        return "unknown"
    if htf == "neutral" or osc_dir == "neutral":
        return "unclear"
    if htf == osc_dir:
        return "confirms"
    return "conflicts"


def market_condition_label(
    *,
    alignment: str,
    htf_trend_dir: str,
    htf_trend_strength: str,
    channel_position: str,
    channel_slope: str,
) -> str:
    req = (alignment, htf_trend_dir, htf_trend_strength, channel_position, channel_slope)
    if any(x == "unknown" for x in req):
        return "unknown"
    # (2) trend
    if (
        alignment == "aligned"
        and htf_trend_dir in ("bullish", "bearish")
        and htf_trend_strength in ("medium", "strong")
    ):
        if htf_trend_dir == "bullish" and (
            channel_slope == "rising" or channel_position == "above"
        ):
            return "trend"
        if htf_trend_dir == "bearish" and (
            channel_slope == "falling" or channel_position == "below"
        ):
            return "trend"
    # (3) range
    if (
        channel_position == "inside"
        and channel_slope == "flat"
        and htf_trend_strength in ("weak", "none")
    ):
        return "range_consolidation"
    # (4) transition
    if alignment == "conflicting" and htf_trend_strength in ("weak", "none"):
        return "transition"
    return "uncertain"


def pullback_state_label(pullback: Optional[bool], scale_conflict: Optional[bool]) -> str:
    if pullback is None:
        return "unknown"
    if not pullback:
        return "no_pullback"
    if scale_conflict:
        return "pullback_with_scale_conflict"
    return "pullback_active"


def scale_conflict_label(scale_conflict: Optional[bool]) -> str:
    if scale_conflict is None:
        return "unknown"
    return "yes" if scale_conflict else "no"


def spread_condition_label(spread_t: float, spreads_through_t: Sequence[float]) -> str:
    if len(spreads_through_t) < 30 or not math.isfinite(spread_t):
        return "unknown"
    arr = [float(x) for x in spreads_through_t if math.isfinite(float(x))]
    if len(arr) < 30:
        return "unknown"
    med = float(sorted(arr)[len(arr) // 2])
    if med <= 0 or not math.isfinite(med):
        return "unknown"
    if spread_t > 2.0 * med:
        return "wide"
    return "normal"


def remaining_opportunity_est_label(
    *,
    target_pct: float,
    equity_pct: float,
    session_phase: float,
    range_so_far: float,
    close_t: float,
) -> str:
    if not all(
        math.isfinite(x)
        for x in (target_pct, equity_pct, session_phase, range_so_far, close_t)
    ):
        return "unknown"
    target_remaining = float(target_pct) - float(equity_pct)
    if target_remaining <= 0:
        return "high"
    time_left = max(0.0, min(1.0, 1.0 - float(session_phase)))
    if close_t <= 0:
        return "unknown"
    range_so_far_pct = 100.0 * float(range_so_far) / float(close_t)
    if time_left < 0.15 and target_remaining > 0.5 * range_so_far_pct:
        return "low"
    if time_left >= 0.50 and target_remaining <= range_so_far_pct:
        return "high"
    return "medium"


def build_agreement_profile(
    *,
    htf_trend_dir: str,
    osc_dir: str,
    channel_position: str,
    spread_condition: str,
    heat_ok: Optional[bool],
) -> Dict[str, Any]:
    trend_group = (
        htf_trend_dir
        if htf_trend_dir in ("bullish", "bearish", "neutral")
        else "unavailable"
    )
    momentum_group = (
        osc_dir if osc_dir in ("bullish", "bearish", "neutral") else "unavailable"
    )
    if channel_position == "above":
        channel_group = "bullish"
    elif channel_position == "below":
        channel_group = "bearish"
    elif channel_position == "inside":
        channel_group = "ranging"
    else:
        channel_group = "unavailable"
    if spread_condition == "normal":
        conditions_group = "normal"
    elif spread_condition == "wide":
        conditions_group = "impaired"
    else:
        conditions_group = "unavailable"
    if heat_ok is True:
        risk_group = "can_act"
    elif heat_ok is False:
        risk_group = "refuse"
    else:
        risk_group = "unavailable"

    groups = {
        "trend": trend_group,
        "momentum": momentum_group,
        "channel": channel_group,
        "conditions": conditions_group,
        "risk": risk_group,
    }
    # Pair agree/conflict only for bullish/bearish directional groups.
    dir_keys = ("trend", "momentum", "channel")
    pairs_agree: List[str] = []
    pairs_conflict: List[str] = []
    for i, a in enumerate(dir_keys):
        for b in dir_keys[i + 1 :]:
            ga, gb = groups[a], groups[b]
            if ga not in ("bullish", "bearish") or gb not in ("bullish", "bearish"):
                continue
            name = f"{a}-{b}"
            if ga == gb:
                pairs_agree.append(name)
            else:
                pairs_conflict.append(name)

    n_avail = sum(1 for v in groups.values() if v != "unavailable")
    return {
        "trend_group": trend_group,
        "momentum_group": momentum_group,
        "channel_group": channel_group,
        "conditions_group": conditions_group,
        "risk_group": risk_group,
        "pairs_agree": pairs_agree,
        "pairs_conflict": pairs_conflict,
        "n_groups_available": n_avail,
    }


def position_label(side: Optional[int]) -> str:
    if side is None:
        return "flat"
    if side > 0:
        return "long"
    if side < 0:
        return "short"
    return "flat"


def heat_ok_estimate(
    *,
    equity_pct: float,
    risk_pct: float,
    banked: bool,
    dead: bool,
) -> bool:
    if banked or dead:
        return False
    heat_dist = max(0.0, (equity_pct - (-risk_pct)) / 100.0)
    return heat_dist > 1e-8


def entry_reason_code(
    *,
    action: int,
    position_before: str,
    htf_trend_dir: str,
    ltf_trend_dir: str,
    heat_ok: bool,
    opened: bool,
    reversed: bool,
) -> str:
    a = int(action)
    if a not in (ACTION_BUY, ACTION_SELL):
        return "none"
    if reversed:
        if a == ACTION_BUY:
            return "entry_reverse_to_bull" if heat_ok else "entry_refused_heat"
        return "entry_reverse_to_bear" if heat_ok else "entry_refused_heat"
    if position_before != "flat":
        return "none"
    if not heat_ok or not opened:
        if a in (ACTION_BUY, ACTION_SELL) and not heat_ok:
            return "entry_refused_heat"
        return "entry_unknown" if a in (ACTION_BUY, ACTION_SELL) else "none"
    if a == ACTION_BUY:
        if htf_trend_dir == "bullish" and heat_ok:
            return "entry_htf_bull"
        if htf_trend_dir == "neutral" and ltf_trend_dir == "bullish" and heat_ok:
            return "entry_ltf_fallback_bull"
        return "entry_unknown"
    if a == ACTION_SELL:
        if htf_trend_dir == "bearish" and heat_ok:
            return "entry_htf_bear"
        if htf_trend_dir == "neutral" and ltf_trend_dir == "bearish" and heat_ok:
            return "entry_ltf_fallback_bear"
        return "entry_unknown"
    return "entry_unknown"


def no_trade_reason_code(
    *,
    action: int,
    position_before: str,
    banked: bool,
    dead: bool,
    breached: bool,
    heat_ok: bool,
    opened: bool,
    reversed: bool,
    exited: bool,
) -> str:
    if opened or reversed or exited:
        return "none"
    if banked:
        return "no_trade_banked"
    if dead or breached:
        return "no_trade_dead_or_breach"
    a = int(action)
    if a in (ACTION_BUY, ACTION_SELL) and not heat_ok:
        return "no_trade_heat_refuse"
    if position_before == "flat" and a == ACTION_HOLD:
        return "no_trade_signal_hold"
    if position_before != "flat" and a == ACTION_HOLD:
        return "no_trade_manage_hold"
    if a in (ACTION_BUY, ACTION_SELL) and position_before == "flat" and not opened:
        return "no_trade_heat_refuse" if not heat_ok else "no_trade_unknown"
    return "no_trade_unknown"


def exit_reason_code(
    *,
    breached_this: bool,
    banked_this: bool,
    stop_this: bool,
    reverse_this: bool,
    eod_flatten: bool = False,
) -> str:
    # Priority: breach > bank > stop > reverse > eod_flatten
    if breached_this:
        return "breach"
    if banked_this:
        return "bank"
    if stop_this:
        return "stop"
    if reverse_this:
        return "reverse"
    if eod_flatten:
        return "eod_flatten"
    return "none"


def day_activity_class_label(
    *,
    cleared: bool,
    breached: bool,
    n_entries: int,
    reversal_count: int,
) -> str:
    if breached:
        return "breached"
    if cleared:
        return "cleared"
    activity = int(n_entries) + int(reversal_count)
    if activity <= 4:
        return "miss_low_activity"
    if activity >= 10:
        return "miss_high_activity"
    return "unknown"


def empty_decision_row() -> Dict[str, Any]:
    return {k: None for k in DECISION_FIELD_NAMES}


def empty_eod_row() -> Dict[str, Any]:
    return {k: None for k in EOD_FIELD_NAMES}


def assert_decision_schema(row: Mapping[str, Any]) -> None:
    missing = [k for k in DECISION_FIELD_NAMES if k not in row]
    if missing:
        raise AssertionError(f"decision row missing fields: {missing}")


def assert_eod_schema(row: Mapping[str, Any]) -> None:
    missing = [k for k in EOD_FIELD_NAMES if k not in row]
    if missing:
        raise AssertionError(f"eod row missing fields: {missing}")


def decision_columns() -> List[str]:
    return list(DECISION_FIELD_NAMES)


def eod_columns() -> List[str]:
    return list(EOD_FIELD_NAMES)
