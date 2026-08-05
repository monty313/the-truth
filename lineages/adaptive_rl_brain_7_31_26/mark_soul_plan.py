"""Mark soul plan teacher — full chart, goal size, force-aligned adds.

When the day chart is known (curriculum diary / BC labels), Mark does not thrash
walk fixed dials. He picks force-aligned entries, sizes to the day's goal, and
adds once if needed — same search that proved 10/10 flexible wins.

Online policy cannot peek the whole day; BC transfers these sparse plans into
weights. Runtime shell still uses goal-relative size + force adds.

Not the banned trail+cushion+scale-in package.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
)

NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

SIZE_GRID = [
    (0.25, 0.20),
    (0.35, 0.25),
    (0.50, 0.35),
    (0.65, 0.45),
    (0.80, 0.55),
    (1.00, 0.70),
]


def force_map(m1, date: str, target: float, risk: float) -> Dict[int, int]:
    day = GoalEquityDay(
        m1,
        target_pct=target,
        risk_pct=risk,
        date_str=date,
        eyes_mode="mark_doctrine",
        mark_soul=False,
    )
    out: Dict[int, int] = {}
    for t in day.runner.decision_indices():
        try:
            out[int(t)] = int(day.recommended_action(t))
        except Exception:
            out[int(t)] = ACTION_HOLD
    return out


def run_plan(
    m1,
    date: str,
    target: float,
    risk: float,
    *,
    risk_use_frac: float,
    per_trade_cap_pct: float,
    plan: Dict[int, int],
) -> Dict[str, Any]:
    """Execute sparse plan with fixed size dials + soul adds (same-side)."""
    day = GoalEquityDay(
        m1,
        target_pct=target,
        risk_pct=risk,
        date_str=date,
        eyes_mode="mark_doctrine",
        risk_use_frac=risk_use_frac,
        per_trade_cap_pct=per_trade_cap_pct,
        mark_soul=True,  # allow force-aligned adds when plan fires same side
    )
    # lock size to plan dials (not dynamic thrash size)
    day._plan_lock_ruf = float(risk_use_frac)
    day._plan_lock_cap = float(per_trade_cap_pct)
    indices = day.runner.decision_indices()
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
        a = int(plan.get(int(t), ACTION_HOLD))
        day.step_action(t, a)
    if not day.dead and not day.banked:
        for bt in range(prev_t, len(day.m1)):
            if day.dead or day.banked:
                break
            day._mark_bar(bt)
    t_last = len(day.m1) - 1
    day._flatten(float(day._close[t_last]), float(day._spread_px[t_last]))
    pnl = 100.0 * (day.balance - day.eq0) / day.eq0
    day.min_eq_pct = min(day.min_eq_pct, pnl)
    if pnl <= -day.risk + 1e-12:
        day.breached = True
    cleared = (pnl >= day.target - 1e-12 and not day.breached) or (
        day.banked and not day.breached
    )
    return {
        "cleared": bool(cleared),
        "breached": bool(day.breached),
        "pnl_pct": round(float(pnl), 4),
        "n_entries": int(day.n_entries),
        "n_adds": int(day.n_adds),
        "min_eq_pct": round(float(day.min_eq_pct), 4),
        "banked": bool(day.banked),
        "risk_use_frac": risk_use_frac,
        "per_trade_cap_pct": per_trade_cap_pct,
        "day": day,
    }


def search_mark_soul_plan(
    m1,
    date: str,
    target: float,
    risk: float,
    *,
    require_force: bool = True,
    max_entry_samples: int = 0,
) -> Dict[str, Any]:
    """Find first force-aligned win: size grid × entry × optional one add.

    max_entry_samples: 0 = all decision bars; else subsample for speed.
    """
    force = force_map(m1, date, target, risk)
    day0 = GoalEquityDay(
        m1,
        target_pct=target,
        risk_pct=risk,
        date_str=date,
        eyes_mode="mark_doctrine",
        mark_soul=False,
    )
    indices = list(day0.runner.decision_indices())
    if max_entry_samples and max_entry_samples < len(indices):
        step = max(1, len(indices) // max_entry_samples)
        entry_ts = indices[::step]
    else:
        entry_ts = indices

    for ruf, cap in SIZE_GRID:
        for i, t1 in enumerate(entry_ts):
            for side in (ACTION_BUY, ACTION_SELL):
                f1 = force.get(int(t1), ACTION_HOLD)
                if require_force and f1 != side:
                    continue
                plan = {int(tt): ACTION_HOLD for tt in indices}
                plan[int(t1)] = side
                res = run_plan(
                    m1,
                    date,
                    target,
                    risk,
                    risk_use_frac=ruf,
                    per_trade_cap_pct=cap,
                    plan=plan,
                )
                if res["cleared"] and not res["breached"]:
                    return {
                        "winnable": True,
                        "best": {
                            **{k: v for k, v in res.items() if k != "day"},
                            "mode": "single",
                            "t1": int(t1),
                            "side": NAMES[side],
                            "plan": plan,
                            "indices": indices,
                        },
                    }
                for t2 in entry_ts[i + 1 :]:
                    f2 = force.get(int(t2), ACTION_HOLD)
                    if require_force and f2 not in (side, ACTION_HOLD):
                        if f2 != side:
                            continue
                    plan2 = {int(tt): ACTION_HOLD for tt in indices}
                    plan2[int(t1)] = side
                    plan2[int(t2)] = side
                    res2 = run_plan(
                        m1,
                        date,
                        target,
                        risk,
                        risk_use_frac=ruf,
                        per_trade_cap_pct=cap,
                        plan=plan2,
                    )
                    if res2["cleared"] and not res2["breached"]:
                        return {
                            "winnable": True,
                            "best": {
                                **{k: v for k, v in res2.items() if k != "day"},
                                "mode": "entry_plus_add",
                                "t1": int(t1),
                                "t2": int(t2),
                                "side": NAMES[side],
                                "plan": plan2,
                                "indices": indices,
                            },
                        }
    return {"winnable": False, "best": None}


def execute_mark_soul_day(
    m1,
    date: str,
    target: float,
    risk: float,
    *,
    max_entry_samples: int = 0,
) -> Dict[str, Any]:
    """Full-chart Mark soul: plan if winnable, else online soul walk."""
    found = search_mark_soul_plan(
        m1,
        date,
        target,
        risk,
        require_force=True,
        max_entry_samples=max_entry_samples,
    )
    if found.get("winnable") and found.get("best"):
        best = found["best"]
        res = run_plan(
            m1,
            date,
            target,
            risk,
            risk_use_frac=float(best["risk_use_frac"]),
            per_trade_cap_pct=float(best["per_trade_cap_pct"]),
            plan=best["plan"],
        )
        day = res.pop("day")
        return {
            "source": "soul_plan",
            "mode": best.get("mode"),
            "cleared": res["cleared"],
            "breached": res["breached"],
            "pnl_pct": res["pnl_pct"],
            "n_entries": res["n_entries"],
            "n_adds": res["n_adds"],
            "min_eq_pct": res["min_eq_pct"],
            "banked": res["banked"],
            "risk_use_frac": res["risk_use_frac"],
            "per_trade_cap_pct": res["per_trade_cap_pct"],
            "plan": best["plan"],
            "day": day,
            "t1": best.get("t1"),
            "t2": best.get("t2"),
            "side": best.get("side"),
        }
    # fallback online soul walk
    day = GoalEquityDay(
        m1,
        target_pct=target,
        risk_pct=risk,
        date_str=date,
        eyes_mode="mark_doctrine",
        mark_soul=True,
    )
    r = day.run(use_heuristic=True)
    return {
        "source": "soul_online_fallback",
        "mode": "walk",
        "cleared": bool(r.cleared),
        "breached": bool(r.breached),
        "pnl_pct": round(float(r.pnl_pct), 4),
        "n_entries": int(r.n_entries),
        "n_adds": int(day.n_adds),
        "min_eq_pct": round(float(r.min_eq_pct), 4),
        "banked": bool(r.banked),
        "risk_use_frac": "dynamic",
        "per_trade_cap_pct": "dynamic",
        "plan": None,
        "day": day,
    }


def collect_soul_plan_labels(
    days: List[Tuple[str, Any]],
    *,
    target: float = 2.0,
    risk: float = 3.0,
    max_days: int = 50,
    multi_pair: bool = True,
    seed: int = 42,
    pairs: Optional[List[Tuple[float, float]]] = None,
    max_entry_samples: int = 36,
    full_obs: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[str, Any]]:
    """(obs, action) from Mark soul plans — sparse entries, force-aligned."""
    from lineages.adaptive_rl_brain_7_31_26.perception.observation import CHANNEL1_DIM
    from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import (
        MARK_FULL_DIM,
    )

    if pairs is None:
        pairs = [(1.0, 2.0), (1.5, 2.5), (2.0, 3.0), (2.5, 3.5), (3.0, 3.5)]
    rng = np.random.default_rng(seed)
    xs: List[np.ndarray] = []
    ys: List[int] = []
    counts = {ACTION_HOLD: 0, ACTION_BUY: 0, ACTION_SELL: 0}
    obs_dim = MARK_FULL_DIM if full_obs else CHANNEL1_DIM
    meta = {
        "n_plan_wins": 0,
        "n_fallback": 0,
        "n_days": 0,
        "max_entry_samples": max_entry_samples,
        "full_obs": bool(full_obs),
        "obs_dim": int(obs_dim),
    }

    for date_str, m1 in days[:max_days]:
        if multi_pair:
            t, r = pairs[int(rng.integers(0, len(pairs)))]
        else:
            t, r = target, risk
        meta["n_days"] += 1
        out = execute_mark_soul_day(
            m1,
            str(date_str),
            t,
            r,
            max_entry_samples=max_entry_samples,
        )
        print(
            f"    soul labels {date_str} T/R={t}/{r} "
            f"src={out['source']} clear={int(out['cleared'])}",
            flush=True,
        )
        day: GoalEquityDay = out["day"]
        plan = out.get("plan")
        if out["source"] == "soul_plan" and plan is not None:
            meta["n_plan_wins"] += 1
            # re-walk with plan, recording labels
            day2 = GoalEquityDay(
                m1,
                target_pct=t,
                risk_pct=r,
                date_str=str(date_str),
                eyes_mode="mark_doctrine",
                risk_use_frac=float(out["risk_use_frac"]),
                per_trade_cap_pct=float(out["per_trade_cap_pct"]),
                mark_soul=True,
                full_obs=full_obs,
            )
            day2._plan_lock_ruf = float(out["risk_use_frac"])
            day2._plan_lock_cap = float(out["per_trade_cap_pct"])
            indices = day2.runner.decision_indices()
            prev_t = 0
            for tb in indices:
                if day2.dead or day2.banked:
                    break
                for bt in range(prev_t, tb):
                    if day2.dead or day2.banked:
                        break
                    day2._mark_bar(bt)
                prev_t = tb + 1
                if day2.dead or day2.banked:
                    break
                obs = day2.observe(tb)
                act = int(plan.get(int(tb), ACTION_HOLD))
                xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
                ys.append(act)
                counts[act] = counts.get(act, 0) + 1
                day2.step_action(tb, act)
        else:
            meta["n_fallback"] += 1
            # online soul labels from fallback walk
            day3 = GoalEquityDay(
                m1,
                target_pct=t,
                risk_pct=r,
                date_str=str(date_str),
                eyes_mode="mark_doctrine",
                mark_soul=True,
                full_obs=full_obs,
            )
            for tb in day3.runner.decision_indices():
                if day3.banked or day3.dead:
                    break
                obs = day3.observe(tb)
                act = int(day3.recommended_action(tb))
                xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
                ys.append(act)
                counts[act] = counts.get(act, 0) + 1
                day3.step_action(tb, act)

    if not xs:
        return (
            np.zeros((0, obs_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            counts,
            meta,
        )
    return (
        np.stack(xs, axis=0),
        np.asarray(ys, dtype=np.int64),
        counts,
        meta,
    )
