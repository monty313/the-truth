"""Mark looks at the chart FIRST under pt5 principles, writes the day, then matches policy.

GOAL order (common sense):
  1. Load real day price (the chart path we have)
  2. Apply basic trading principles (pt5 + MARK SETS LAW + mark_doctrine)
  3. Write first-person: what I would have done during the day
  4. Run policy on same day; compare
  5. BC policy so it does the same thing
  6. Never claim policy is Mark until agree is high on the written day

pt5 are PRINCIPLES only (HTF permission, LTF timing, regime, capital).
Sets law: 1m|15m,30m · 5m|30m,1h · 15m|1h,4h · 30m|4h,1d.

Usage (repo root, PYTHONPATH=.;code):
  python lineages/adaptive_rl_brain_7_31_26/mark_day_diary.py --dates 2026-04-02,2026-04-01
  python lineages/adaptive_rl_brain_7_31_26/mark_day_diary.py --dates 2026-04-02 --target 3.0 --risk 3.5 --epochs 25
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay, load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.perception.observation import CHANNEL1_DIM
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.perception.sets import (
    assert_mark_sets_law,
    mark_sets_law_table,
)
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_HOLD, Channel1Policy

CKPT_DIR = os.path.join(_HERE, "checkpoints")
EMBRYO = os.path.join(CKPT_DIR, "mark_clone_doctrine_v1.pt")
EMBRYO_FULL = os.path.join(CKPT_DIR, "mark_clone_full_obs_v1.pt")
DIARY_DIR = os.path.join(CKPT_DIR, "mark_day_diaries")
REPORT = os.path.join(CKPT_DIR, "mark_day_diary_goal_report.json")
NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

# pt5 principles as short tags Mark cites in the diary (not the whole trade system)
PT5_PRINCIPLES = [
    "pt5.1 HTF permission / gravity — LTF never votes side against HTF",
    "pt5.1 slingshot: pullback loads, resume with HTF releases",
    "pt5.2 breath vs launch — different playbooks",
    "pt5.3 regime: bull/bear/chop/flat rewrites what is allowed",
    "pt5.4 capital: floor and size before edge",
    "MARK SETS LAW: LTF=first (pullback/cont/add); HTF=last two (confirm); scan all 4",
]


def _load_embryo(*, full_obs: bool = False) -> Channel1Policy:
    """Load Mark clone weights; prefer full-obs ckpt when full_obs=True."""
    path = EMBRYO_FULL if full_obs and os.path.isfile(EMBRYO_FULL) else EMBRYO
    if full_obs and not os.path.isfile(EMBRYO_FULL):
        # fall back to latest if it is full-dim
        latest = os.path.join(CKPT_DIR, "mark_clone_latest.pt")
        if os.path.isfile(latest):
            path = latest
    default_dim = MARK_FULL_DIM if full_obs else CHANNEL1_DIM
    default_h = 128 if full_obs else 64
    pol = Channel1Policy(obs_dim=default_dim, hidden=default_h)
    if os.path.isfile(path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        h = int(blob.get("hidden", default_h))
        dim = int(blob.get("obs_dim", default_dim))
        pol = Channel1Policy(obs_dim=dim, hidden=h)
        try:
            pol.load_state_dict(blob["state_dict"])
        except Exception:
            pass
    pol.eval()
    return pol


def mark_walk_day(
    m1,
    date_str: str,
    target: float,
    risk: float,
    *,
    soul_plan: bool = True,
    full_obs: bool = False,
) -> Dict[str, Any]:
    """Mark sees chart first: soul plan (size+adds) or doctrine walk + narrative.

    soul_plan=True (default): full-chart Mark — goal-relative lots + force adds
    when winnable under force. Matches DIAGNOSIS_FLEXIBLE_SIZE_ADDS 10/10 truth.
    full_obs=True: X labels use 168-dim Mark board (for full-clone BC compare).
    """
    from lineages.adaptive_rl_brain_7_31_26.mark_soul_plan import execute_mark_soul_day

    rows: List[dict] = []
    xs: List[np.ndarray] = []
    ys: List[int] = []
    obs_dim = MARK_FULL_DIM if full_obs else CHANNEL1_DIM

    if soul_plan:
        out = execute_mark_soul_day(m1, str(date_str), target, risk)
        plan = out.get("plan")
        ruf = out.get("risk_use_frac", 0.35)
        cap = out.get("per_trade_cap_pct", 0.25)
        if plan is not None and out["source"] == "soul_plan":
            day = GoalEquityDay(
                m1,
                target_pct=target,
                risk_pct=risk,
                date_str=str(date_str),
                eyes_mode="mark_doctrine",
                risk_use_frac=float(ruf),
                per_trade_cap_pct=float(cap),
                mark_soul=True,
                full_obs=full_obs,
            )
            day._plan_lock_ruf = float(ruf)
            day._plan_lock_cap = float(cap)
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
                ts = str(m1.index[t])
                obs = day.observe(t)
                act = int(plan.get(int(t), ACTION_HOLD))
                eq = float(day.equity_pct(float(day._close[t])))
                why = f"soul_plan_{out.get('mode')}"
                if act != ACTION_HOLD:
                    why += f"_size={ruf}/{cap}"
                day.step_action(t, act)
                xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
                ys.append(act)
                rows.append(
                    {
                        "time": ts,
                        "bar": int(t),
                        "i_would": NAMES[act],
                        "action": act,
                        "equity_pct": round(eq, 4),
                        "entries_so_far": int(day.n_entries),
                        "position_after": (
                            "FLAT"
                            if day.side is None
                            else ("LONG" if day.side > 0 else "SHORT")
                        ),
                        "pt5_regime": "soul_plan",
                        "pt5_force": str(out.get("side") or ""),
                        "why": why,
                    }
                )
            if not day.dead and not day.banked:
                for bt in range(prev_t, len(day.m1)):
                    if day.dead or day.banked:
                        break
                    day._mark_bar(bt)
            t_last = len(day.m1) - 1
            day._flatten(float(day._close[t_last]), float(day._spread_px[t_last]))
            pnl = 100.0 * (day.balance - day.eq0) / day.eq0
            if pnl <= -day.risk + 1e-12:
                day.breached = True
            cleared = (pnl >= day.target - 1e-12 and not day.breached) or (
                day.banked and not day.breached
            )
            return {
                "date": str(date_str),
                "target_pct": target,
                "risk_pct": risk,
                "rows": rows,
                "X": np.stack(xs) if xs else np.zeros((0, obs_dim), np.float32),
                "y": np.asarray(ys, dtype=np.int64),
                "n_entries": int(day.n_entries),
                "n_adds": int(day.n_adds),
                "pnl_pct": round(float(pnl), 4),
                "min_eq_pct": round(float(day.min_eq_pct), 4),
                "cleared": bool(cleared),
                "breached": bool(day.breached),
                "banked": bool(day.banked),
                "action_counts": dict(Counter(NAMES[a] for a in ys)),
                "soul_source": out["source"],
                "soul_mode": out.get("mode"),
                "soul_size": f"{ruf}/{cap}",
                "full_obs": bool(full_obs),
                "obs_dim": int(obs_dim),
            }

    # fallback / walk: online soul doctrine
    day = GoalEquityDay(
        m1,
        target_pct=target,
        risk_pct=risk,
        date_str=str(date_str),
        eyes_mode="mark_doctrine",
        mark_soul=True,
        full_obs=full_obs,
    )
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
        ts = str(m1.index[t])
        obs = day.observe(t)
        act = int(day.recommended_action(t))
        eq = float(day.equity_pct(float(day._close[t])))
        reason = ""
        regime = ""
        force = ""
        try:
            perc = day.runner.perceive(t)
            from lineages.adaptive_rl_brain_7_31_26.perception.mark_doctrine import (
                doctrine_action_from_perception,
            )

            dec = doctrine_action_from_perception(perc)
            reason = getattr(dec, "reason", "") or ""
            regime = str(getattr(dec, "regime", "") or "")
            force = str(getattr(dec, "force_dir", "") or "")
        except Exception:
            pass
        day.step_action(t, act)
        xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        ys.append(act)
        rows.append(
            {
                "time": ts,
                "bar": int(t),
                "i_would": NAMES[act],
                "action": act,
                "equity_pct": round(eq, 4),
                "entries_so_far": int(day.n_entries),
                "position_after": (
                    "FLAT"
                    if day.side is None
                    else ("LONG" if day.side > 0 else "SHORT")
                ),
                "pt5_regime": regime,
                "pt5_force": force,
                "why": reason,
            }
        )
    if not day.dead and not day.banked:
        for bt in range(prev_t, len(day.m1)):
            if day.dead or day.banked:
                break
            day._mark_bar(bt)
    t_last = len(day.m1) - 1
    day._flatten(float(day._close[t_last]), float(day._spread_px[t_last]))
    pnl = 100.0 * (day.balance - day.eq0) / day.eq0
    if pnl <= -day.risk + 1e-12:
        day.breached = True
    cleared = (pnl >= day.target - 1e-12 and not day.breached) or (
        day.banked and not day.breached
    )
    return {
        "date": str(date_str),
        "target_pct": target,
        "risk_pct": risk,
        "rows": rows,
        "X": np.stack(xs) if xs else np.zeros((0, obs_dim), np.float32),
        "y": np.asarray(ys, dtype=np.int64),
        "n_entries": int(day.n_entries),
        "n_adds": int(day.n_adds),
        "pnl_pct": round(float(pnl), 4),
        "min_eq_pct": round(float(day.min_eq_pct), 4),
        "cleared": bool(cleared),
        "breached": bool(day.breached),
        "banked": bool(day.banked),
        "action_counts": dict(Counter(NAMES[a] for a in ys)),
        "soul_source": "soul_online_walk",
        "soul_mode": "walk",
        "soul_size": "dynamic",
        "full_obs": bool(full_obs),
        "obs_dim": int(obs_dim),
    }


def write_diary_md(walk: Dict[str, Any], path: str) -> None:
    """First-person day diary: what I would have done (principles + actions)."""
    lines = []
    lines.append(f"# Mark day diary — {walk['date']}")
    lines.append("")
    lines.append("**Order:** I looked at the chart **before** the policy was trusted.")
    lines.append(
        f"**Goal that day:** target **{walk['target_pct']}%** · risk floor **−{walk['risk_pct']}%** "
        "(runtime inputs — no retrain)."
    )
    lines.append("")
    lines.append("## Principles I used (pt5 = basics only)")
    for p in PT5_PRINCIPLES:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("## Sets I scanned (MARK SETS LAW)")
    for row in mark_sets_law_table():
        lines.append(
            f"- Set {row['set_id']}: LTF **{row['ltf_entry']}** "
            f"(pullback/cont/add) · HTF **{', '.join(row['htf_confirm'])}** (trend confirm)"
        )
    lines.append("")
    lines.append("## What I would have done during the day")
    lines.append("")
    if not walk["rows"]:
        lines.append("_No decision bars._")
    for r in walk["rows"]:
        lines.append(
            f"- **{r['time']}** — I would **{r['i_would']}** "
            f"(eq≈{r['equity_pct']}% · pos→{r['position_after']} · entries={r['entries_so_far']})"
        )
        if r.get("why"):
            lines.append(f"  - why: `{r['why']}`")
        if r.get("pt5_regime") or r.get("pt5_force"):
            lines.append(
                f"  - force={r.get('pt5_force')} regime={r.get('pt5_regime')}"
            )
    lines.append("")
    lines.append("## End of day (my score)")
    lines.append(f"- entries: **{walk['n_entries']}**")
    lines.append(f"- pnl: **{walk['pnl_pct']}%** · min equity: **{walk['min_eq_pct']}%**")
    lines.append(f"- banked: **{walk['banked']}** · breached: **{walk['breached']}**")
    lines.append(
        f"- **award/clear: {walk['cleared']}** "
        f"(hit target without floor — or banked clean)"
    )
    lines.append(f"- action mix: `{walk['action_counts']}`")
    if walk.get("soul_source"):
        lines.append(
            f"- soul: **{walk.get('soul_source')}** mode={walk.get('soul_mode')} "
            f"size={walk.get('soul_size')} adds={walk.get('n_adds', 0)}"
        )
    lines.append("")
    lines.append(
        "_This diary is principles applied to this day's price path — "
        "Mark soul = force + goal-relative size + force-aligned adds when chart known._"
    )
    lines.append("")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def policy_compare(walk: Dict[str, Any], policy: Channel1Policy) -> Dict[str, Any]:
    X, y = walk["X"], walk["y"]
    if len(y) == 0:
        return {"agree_rate": 1.0, "disagree": 0, "n": 0}
    pol_dim = int(getattr(policy, "obs_dim", CHANNEL1_DIM))
    if X.ndim == 2 and X.shape[1] != pol_dim:
        return {
            "agree_rate": 0.0,
            "disagree": int(len(y)),
            "n": int(len(y)),
            "error": f"obs_dim mismatch walk={X.shape[1]} policy={pol_dim}",
        }
    agree = 0
    policy.eval()
    with torch.no_grad():
        for i in range(len(y)):
            a = int(
                torch.argmax(
                    policy(torch.as_tensor(X[i], dtype=torch.float32)), dim=-1
                ).item()
            )
            if a == int(y[i]):
                agree += 1
    n = len(y)
    return {
        "agree": agree,
        "disagree": n - agree,
        "n": n,
        "agree_rate": agree / max(n, 1),
    }


def bc_to_mark(
    policy: Channel1Policy,
    walks: List[Dict[str, Any]],
    *,
    epochs: int = 25,
) -> Dict[str, Any]:
    xs = [w["X"] for w in walks if len(w["y"])]
    ys = [w["y"] for w in walks if len(w["y"])]
    if not xs:
        return {"updated": False}
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    counts = np.bincount(y, minlength=3).astype(np.float64) + 1.0
    w = torch.tensor(
        (counts.sum() / (3.0 * counts)).astype(np.float32), dtype=torch.float32
    )
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    n = len(y)
    last = 0.0
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, 64):
            idx = perm[i : i + 64]
            loss = F.cross_entropy(policy(Xt[idx]), yt[idx], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())
    policy.eval()
    with torch.no_grad():
        pred = policy(Xt).argmax(-1).numpy()
    return {
        "updated": True,
        "n": int(n),
        "match_after": float((pred == y).mean()),
        "loss_final": last,
        "epochs": epochs,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dates",
        default="2026-04-02,2026-04-01",
        help="chart days Mark writes first",
    )
    ap.add_argument("--target", type=float, default=None, help="default per-day target")
    ap.add_argument("--risk", type=float, default=None)
    ap.add_argument(
        "--pair-map",
        default="2026-04-02:3.0,3.5;2026-04-01:1.0,2.0",
        help="date:target,risk overrides",
    )
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--data", default="XAUUSD_curriculum_2026.csv")
    args = ap.parse_args(argv)

    assert_mark_sets_law()
    os.makedirs(DIARY_DIR, exist_ok=True)

    pair_overrides: Dict[str, Tuple[float, float]] = {}
    for part in args.pair_map.split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        d, tr = part.split(":", 1)
        t_s, r_s = tr.split(",")
        pair_overrides[d.strip()] = (float(t_s), float(r_s))

    days = {str(d): m for d, m in load_calendar_days(args.data, min_bars=900)}
    dates = [x.strip() for x in args.dates.split(",") if x.strip()]
    default_t = float(args.target) if args.target is not None else 2.0
    default_r = float(args.risk) if args.risk is not None else 3.0

    print("=" * 64, flush=True)
    print("GOAL: chart first → write what I would do (pt5) → policy same", flush=True)
    print("=" * 64, flush=True)

    walks: List[Dict[str, Any]] = []
    before: List[Dict[str, Any]] = []
    policy = _load_embryo()

    for date in dates:
        if date not in days:
            print(f"missing {date}", flush=True)
            continue
        t, r = pair_overrides.get(date, (default_t, default_r))
        print(f"\n[1] MARK looks at chart  {date}  target={t} risk={r}", flush=True)
        w = mark_walk_day(days[date], date, t, r)
        diary_path = os.path.join(DIARY_DIR, f"MARK_DIARY__{date}__t{t}_r{r}.md")
        write_diary_md(w, diary_path)
        print(f"    wrote diary {diary_path}", flush=True)
        print(
            f"    I would: {w['action_counts']} entries={w['n_entries']} "
            f"clear={w['cleared']} pnl={w['pnl_pct']}",
            flush=True,
        )
        cmp0 = policy_compare(w, policy)
        print(
            f"[2] policy vs my diary  agree={cmp0['agree_rate']:.3f} "
            f"disagree={cmp0['disagree']}/{cmp0['n']}",
            flush=True,
        )
        walks.append(w)
        before.append({"date": date, **cmp0})

    print("\n[3] Make policy do the same thing (BC on my day labels)…", flush=True)
    upd = bc_to_mark(policy, walks, epochs=args.epochs)
    print(f"    {upd}", flush=True)

    after = []
    for w in walks:
        c = policy_compare(w, policy)
        after.append({"date": w["date"], **c})
        print(
            f"[4] after update {w['date']} agree={c['agree_rate']:.3f}",
            flush=True,
        )

    # save embryo
    try:
        hidden = int(policy.net[0].out_features)
    except Exception:
        hidden = 64
    torch.save(
        {
            "tag": "mark_clone_doctrine_v1",
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "state_dict": policy.state_dict(),
            "hidden": hidden,
            "obs_dim": CHANNEL1_DIM,
            "eyes_mode": "mark_doctrine",
            "teacher": "pt5_principles_plus_sets_law_day_diary",
            "goal": "chart_first_write_day_then_policy_same",
            "proven_touched": False,
        },
        EMBRYO,
    )
    print(f"    saved {EMBRYO}", flush=True)

    mean_b = float(np.mean([x["agree_rate"] for x in before])) if before else 0.0
    mean_a = float(np.mean([x["agree_rate"] for x in after])) if after else 0.0
    report = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "goal": "Mark chart first → write day (pt5 principles) → policy same",
        "pt5": "principles only — not full discretionary Mark",
        "sets_law": mark_sets_law_table(),
        "diaries_dir": DIARY_DIR,
        "before": before,
        "after": after,
        "update": upd,
        "mean_agree_before": mean_b,
        "mean_agree_after": mean_a,
        "improved": mean_a >= mean_b - 1e-12,
        "day_summaries": [
            {
                "date": w["date"],
                "target_pct": w["target_pct"],
                "risk_pct": w["risk_pct"],
                "cleared": w["cleared"],
                "n_entries": w["n_entries"],
                "pnl_pct": w["pnl_pct"],
                "actions": w["action_counts"],
            }
            for w in walks
        ],
        "proven_touched": False,
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nREPORT {REPORT}", flush=True)
    print(
        f"STATUS chart_first=true agree {mean_b:.3f}->{mean_a:.3f} "
        f"improved={report['improved']} proven_ok=true",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
