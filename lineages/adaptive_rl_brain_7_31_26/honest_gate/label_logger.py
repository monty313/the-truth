"""Label Contract V1 day logger — observation only; shell/heuristic unchanged.

CHANGE LOG:
- 2026-07-31  V1 decision+EOD logger — WHY: Principle 9 explainability for
  multi-pair tutor. Does not change entries, dials, rewards, or PROVEN.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from lineages.adaptive_rl_brain_7_31_26.data.mtf import bar_asof
from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay
from lineages.adaptive_rl_brain_7_31_26.honest_gate.label_contract_v1 import (
    ACTION_NAME,
    LABEL_CONTRACT_VERSION,
    alignment_label,
    build_agreement_profile,
    cci_state_label,
    channel_position_label,
    channel_slope_label,
    day_activity_class_label,
    dir_to_label,
    empty_decision_row,
    empty_eod_row,
    entry_reason_code,
    exit_reason_code,
    heat_ok_estimate,
    market_condition_label,
    momentum_velocity_change_label,
    momentum_vs_direction_label,
    no_trade_reason_code,
    position_label,
    pullback_state_label,
    remaining_opportunity_est_label,
    rsi_state_label,
    scale_conflict_label,
    spread_condition_label,
    velocity_to_label,
)
from lineages.adaptive_rl_brain_7_31_26.perception.confluence import majority_direction
from lineages.adaptive_rl_brain_7_31_26.perception.live_indicators import (
    dual_confirmation_flags,
    snapshot_at,
)
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
)


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _snap_pair(ind: Dict[str, pd.DataFrame], tf: str, ts: pd.Timestamp):
    fr = ind.get(tf)
    if fr is None or len(fr) == 0:
        return None
    i = bar_asof(fr, ts)
    if i < 12:
        return None
    try:
        return snapshot_at(fr, tf, i), i, fr
    except Exception:
        return None


def _channel_mid_at(fr: pd.DataFrame, i: int) -> float:
    try:
        hi = float(fr.iloc[i]["ch_high_s2"])
        lo = float(fr.iloc[i]["ch_low_s2"])
    except Exception:
        return float("nan")
    if not (_finite(hi) and _finite(lo)):
        return float("nan")
    return 0.5 * (hi + lo)


def _osc_dir_from_votes(votes: Sequence[Any]) -> str:
    if not votes:
        return "unknown"
    dirs = [getattr(v, "direction", None) for v in votes]
    if any(d is None for d in dirs):
        return "unknown"
    md = majority_direction(votes)
    return dir_to_label(md)


def _inside_on_tf(snap) -> bool:
    if snap is None:
        return False
    c, hi, lo = snap.close, snap.channel_high, snap.channel_low
    if not (_finite(c) and _finite(hi) and _finite(lo)):
        return False
    return float(lo) <= float(c) <= float(hi)


def compute_pre_action_labels(
    day: GoalEquityDay,
    t: int,
    *,
    meaning_hash: str,
    split: str,
    htf_dir_prev: Optional[str],
    vel_prev: Optional[str],
) -> Dict[str, Any]:
    """Build all before-action labels at decision bar t (no future bars)."""
    runner = day.runner
    n = len(day.m1)
    phase = float(t / max(n - 1, 1))
    price = float(day._close[t])
    eq = float(day.equity_pct(price))
    danger_unused = 0.0  # perception path uses progress/danger in observe only
    _ = danger_unused

    # Perceive with flat side for HTF/LTF eyes (same as recommended_action).
    perc = runner.perceive(t, trade_side=None)
    primary = perc.get("primary")
    higher = perc.get("higher")
    lower = perc.get("lower")
    struct = perc.get("structure")

    htf = dir_to_label(higher)
    ltf = dir_to_label(lower)
    htf_str = velocity_to_label(getattr(primary, "velocity", None) if primary else None)
    mom_vel = htf_str
    mom_vel_chg = (
        momentum_velocity_change_label(mom_vel, vel_prev)
        if vel_prev is not None
        else "unknown"
    )

    # Indicator snapshots for Set-2 confirmation TFs 30m + 1h.
    ts = day.m1.index[t]
    ind = runner.ind
    sa_pack = _snap_pair(ind, "30m", ts)
    sb_pack = _snap_pair(ind, "1h", ts)

    both_above = both_below = False
    inside_both = False
    ready = False
    cci30 = cci100 = rsi5 = rsi14 = float("nan")
    cci30_prev = rsi14_prev = None
    ch_slope = "unknown"
    osc_dir = "unknown"

    if primary is not None and getattr(primary, "votes", None):
        osc_dir = _osc_dir_from_votes(primary.votes)

    if sa_pack is not None and sb_pack is not None:
        sa, ia, fra = sa_pack
        sb, ib, frb = sb_pack
        dual = dual_confirmation_flags(sa, sb)
        both_above, both_below = dual["channel"]
        inside_both = _inside_on_tf(sa) and _inside_on_tf(sb)
        ready = True
        cci30, cci100 = float(sa.cci30), float(sa.cci100)
        rsi5, rsi14 = float(sa.rsi5), float(sa.rsi14)

        # Prior decision bar for slope / strengthening (causal only).
        t_prev = t - int(day.decide_every)
        warm = max(12, int(getattr(runner, "warmup", 12) or 12))
        if t_prev >= warm:
            ts_prev = day.m1.index[t_prev]
            sa_prev_pack = _snap_pair(ind, "30m", ts_prev)
            if sa_prev_pack is not None:
                sa_p, ip, _ = sa_prev_pack
                cci30_prev = float(sa_p.cci30)
                rsi14_prev = float(sa_p.rsi14)
                mid_t = _channel_mid_at(fra, ia)
                mid_p = _channel_mid_at(fra, ip)
                ch_slope = channel_slope_label(mid_t, mid_p)

    ch_pos = channel_position_label(
        both_above=bool(both_above),
        both_below=bool(both_below),
        inside_both=bool(inside_both),
        ready=ready,
    )
    cci_st = cci_state_label(cci30, cci100, cci30_prev)
    rsi_st = rsi_state_label(rsi5, rsi14, rsi14_prev)
    align = alignment_label(htf, ltf)
    mvd = momentum_vs_direction_label(htf, osc_dir)
    mkt = market_condition_label(
        alignment=align,
        htf_trend_dir=htf,
        htf_trend_strength=htf_str,
        channel_position=ch_pos,
        channel_slope=ch_slope,
    )
    pb = getattr(struct, "pullback", None) if struct is not None else None
    sc = getattr(struct, "scale_conflict", None) if struct is not None else None
    pb_state = pullback_state_label(pb if pb is None else bool(pb), sc if sc is None else bool(sc))
    sc_lab = scale_conflict_label(sc if sc is None else bool(sc))

    spreads = day._spread_px[: t + 1]
    sp_t = float(day._spread_px[t])
    sp_cond = spread_condition_label(sp_t, spreads)

    heat_ok = heat_ok_estimate(
        equity_pct=eq,
        risk_pct=float(day.risk),
        banked=bool(day.banked),
        dead=bool(day.dead),
    )
    profile = build_agreement_profile(
        htf_trend_dir=htf,
        osc_dir=osc_dir,
        channel_position=ch_pos,
        spread_condition=sp_cond,
        heat_ok=heat_ok,
    )

    range_so_far = float(np.max(day._high[: t + 1]) - np.min(day._low[: t + 1]))
    rem_opp = remaining_opportunity_est_label(
        target_pct=float(day.target),
        equity_pct=eq,
        session_phase=phase,
        range_so_far=range_so_far,
        close_t=price,
    )

    reversed_while = False
    if htf_dir_prev is not None and htf_dir_prev == htf and htf in ("bullish", "bearish"):
        # filled later if reverse happens; pre-flag only means HTF unchanged
        reversed_while = True  # candidate; AND with reverse event in post

    return {
        "label_contract_version": LABEL_CONTRACT_VERSION,
        "meaning_hash": str(meaning_hash),
        "date": day.date_str,
        "t": int(t),
        "split": str(split),
        "target_pct": float(day.target),
        "risk_pct": float(day.risk),
        "htf_trend_dir": htf,
        "htf_trend_strength": htf_str,
        "ltf_trend_dir": ltf,
        "alignment": align,
        "channel_position": ch_pos,
        "channel_slope": ch_slope,
        "cci_state": cci_st,
        "rsi_state": rsi_st,
        "momentum_velocity": mom_vel,
        "momentum_velocity_change": mom_vel_chg,
        "momentum_vs_direction": mvd,
        "market_condition": mkt,
        "pullback_state": pb_state,
        "scale_conflict": sc_lab,
        "agreement_profile": profile,
        "session_phase": round(phase, 6),
        "equity_pct": round(eq, 6),
        "target_remaining_pct": round(float(day.target) - eq, 6),
        "risk_remaining_pct": round(float(day.risk) + eq, 6),
        "heat_ok": bool(heat_ok),
        "spread_condition": sp_cond,
        "remaining_opportunity_est": rem_opp,
        "position_before": position_label(day.side),
        "_htf_unchanged_candidate": reversed_while,
        "_vel": mom_vel,
        "_htf": htf,
        "future_bar_used": False,
    }


def log_equity_day(
    day: GoalEquityDay,
    *,
    meaning_hash: str,
    split: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run heuristic day with V1 decision rows + EOD row. Policy byte-identical path."""
    decision_rows: List[Dict[str, Any]] = []
    indices = list(day.runner.decision_indices())
    prev_t = 0
    htf_prev: Optional[str] = None
    vel_prev: Optional[str] = None
    reversal_count = 0
    entry_number = 0  # cumulative entries after step (n_entries mirror)

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

        pre = compute_pre_action_labels(
            day,
            t,
            meaning_hash=meaning_hash,
            split=split,
            htf_dir_prev=htf_prev,
            vel_prev=vel_prev,
        )
        # Capture shell state before action
        side_before = day.side
        n_entries_before = int(day.n_entries)
        n_closes_before = int(day.n_closes)
        banked_before = bool(day.banked)
        breached_before = bool(day.breached)
        stop_before = float(day.stop) if day.side is not None else None

        action = int(day.recommended_action(t))
        day.step_action(t, action)

        opened = int(day.n_entries) > n_entries_before and side_before is None
        reversed_open = (
            side_before is not None
            and day.side is not None
            and int(day.n_entries) > n_entries_before
        )
        # reverse: flatten+open opposite, n_entries increases
        if reversed_open:
            reversal_count += 1
        stop_this = (
            side_before is not None
            and day.side is None
            and int(day.n_closes) > n_closes_before
            and not bool(day.banked and not banked_before)
            and not bool(day.breached and not breached_before)
            and not reversed_open
            and action == ACTION_HOLD
        )
        # More reliable stop detection: closes increased, not reverse, not bank, not breach
        if (
            side_before is not None
            and day.side is None
            and int(day.n_closes) > n_closes_before
            and not reversed_open
        ):
            if bool(day.breached) and not breached_before:
                stop_this = False  # breach path
            elif bool(day.banked) and not banked_before:
                stop_this = False
            else:
                # stop or reverse-only flatten handled above
                if action in (ACTION_BUY, ACTION_SELL) and not reversed_open:
                    # directional but closed without re-open → treat as stop if mark did it
                    stop_this = True
                elif action == ACTION_HOLD:
                    stop_this = True

        banked_this = bool(day.banked) and not banked_before
        breached_this = bool(day.breached) and not breached_before
        exited = (
            side_before is not None
            and day.side is None
            and (stop_this or banked_this or breached_this or reversed_open)
        )

        heat_ok = bool(pre["heat_ok"])
        # If open was refused while directional, heat may have been the cause
        if (
            side_before is None
            and action in (ACTION_BUY, ACTION_SELL)
            and day.side is None
            and not heat_ok
        ):
            opened = False

        e_reason = entry_reason_code(
            action=action,
            position_before=str(pre["position_before"]),
            htf_trend_dir=str(pre["htf_trend_dir"]),
            ltf_trend_dir=str(pre["ltf_trend_dir"]),
            heat_ok=heat_ok,
            opened=bool(opened or reversed_open),
            reversed=bool(reversed_open),
        )
        x_reason = exit_reason_code(
            breached_this=breached_this,
            banked_this=banked_this,
            stop_this=bool(stop_this),
            reverse_this=bool(reversed_open),
        )
        nt_reason = no_trade_reason_code(
            action=action,
            position_before=str(pre["position_before"]),
            banked=bool(day.banked),
            dead=bool(day.dead),
            breached=bool(day.breached),
            heat_ok=heat_ok,
            opened=bool(opened),
            reversed=bool(reversed_open),
            exited=bool(exited) or x_reason != "none",
        )

        entry_number = int(day.n_entries)
        rev_while = bool(
            reversed_open and pre.get("_htf_unchanged_candidate") and htf_prev is not None
        )

        row = empty_decision_row()
        for k in row:
            if k in pre and not k.startswith("_"):
                row[k] = pre[k]
        row["action"] = ACTION_NAME.get(action, str(action))
        row["entry_number"] = entry_number
        row["reversal_number"] = int(reversal_count)
        row["reversed_while_htf_unchanged"] = bool(rev_while)
        row["entry_reason"] = e_reason
        row["exit_reason"] = x_reason
        row["no_trade_reason"] = nt_reason
        row["future_bar_used"] = False
        # drop internal keys if any leaked
        row.pop("_htf_unchanged_candidate", None)
        row.pop("_vel", None)
        row.pop("_htf", None)
        decision_rows.append(row)

        htf_prev = str(pre.get("_htf") or pre.get("htf_trend_dir"))
        vel_prev = str(pre.get("_vel") or pre.get("momentum_velocity"))
        _ = stop_before  # silence unused

    # Mark remaining bars to EOD (same as GoalEquityDay.run)
    if not day.dead and not day.banked:
        for bt in range(prev_t, len(day.m1)):
            if day.dead or day.banked:
                break
            day._mark_bar(bt)

    t_last = len(day.m1) - 1
    price = float(day._close[t_last])
    sp = float(day._spread_px[t_last])
    day._flatten(price, sp)
    pnl = 100.0 * (day.balance - day.eq0) / day.eq0
    day.min_eq_pct = min(day.min_eq_pct, pnl)
    if pnl <= -day.risk + 1e-12:
        day.breached = True
    goal_hit = (pnl >= day.target - 1e-12) and (not day.breached)
    if day.banked and not day.breached and pnl >= day.target - 1e-9:
        goal_hit = True

    eod = empty_eod_row()
    eod.update(
        {
            "label_contract_version": LABEL_CONTRACT_VERSION,
            "meaning_hash": str(meaning_hash),
            "date": day.date_str,
            "split": str(split),
            "target_pct": float(day.target),
            "risk_pct": float(day.risk),
            "cleared": bool(goal_hit),
            "breached": bool(day.breached),
            "banked": bool(day.banked),
            "pnl_pct": round(float(pnl), 6),
            "min_eq_pct": round(float(day.min_eq_pct), 6),
            "dist_to_target_pct": round(float(day.target) - float(pnl), 6),
            "n_entries": int(day.n_entries),
            "reversal_count": int(reversal_count),
            "day_activity_class": day_activity_class_label(
                cleared=bool(goal_hit),
                breached=bool(day.breached),
                n_entries=int(day.n_entries),
                reversal_count=int(reversal_count),
            ),
            "n_decision_rows": int(len(decision_rows)),
        }
    )
    return decision_rows, eod


def make_day(
    m1: pd.DataFrame,
    *,
    date_str: str,
    target_pct: float,
    risk_pct: float,
    dials: Dict[str, Any],
) -> GoalEquityDay:
    return GoalEquityDay(
        m1,
        target_pct=float(target_pct),
        risk_pct=float(risk_pct),
        date_str=str(date_str),
        risk_use_frac=float(dials.get("risk_use_frac", 0.35)),
        stop_atr_mult=float(dials.get("stop_atr_mult", 2.0)),
        per_trade_cap_pct=float(dials.get("per_trade_cap_pct", 0.25)),
        decide_every=25,
        use_signal_majority=False,
    )
