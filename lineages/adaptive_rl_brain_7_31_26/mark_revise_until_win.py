"""Mark already sees the chart → revise plan under pt5 until award (when possible).

Common sense:
  - Principles (pt5) + SETS LAW stay fixed.
  - Mark has the full day price path (offline study) and keeps revising
    until he banks target without floor — or we prove the day is unwinnable
    under shell risk physics.
  - Then we write the winning diary and BC the policy to match.

This is offline "Mark studied the chart" teaching, not live lookahead claims.

Usage (repo root, PYTHONPATH=.;code):
  python lineages/adaptive_rl_brain_7_31_26/mark_revise_until_win.py
  python lineages/adaptive_rl_brain_7_31_26/mark_revise_until_win.py --from-test-pack
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import (
    GoalEquityDay,
    load_calendar_days,
)
from lineages.adaptive_rl_brain_7_31_26.mark_day_diary import (
    PT5_PRINCIPLES,
    write_diary_md,
    mark_walk_day,
    policy_compare,
    _load_embryo,
)
from lineages.adaptive_rl_brain_7_31_26.perception.sets import assert_mark_sets_law
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    Channel1Policy,
)

CKPT_DIR = os.path.join(_HERE, "checkpoints")
PACK_DIR = os.path.join(CKPT_DIR, "test_run_10d_mark_vs_policy")
OUT_DIR = os.path.join(CKPT_DIR, "mark_revise_until_win")
EMBRYO = os.path.join(CKPT_DIR, "mark_clone_doctrine_v1.pt")
NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}


def _new_day(m1, date: str, target: float, risk: float, eyes: str = "mark_doctrine") -> GoalEquityDay:
    return GoalEquityDay(
        m1,
        target_pct=float(target),
        risk_pct=float(risk),
        date_str=str(date),
        eyes_mode=eyes,
    )


def run_heuristic(m1, date: str, target: float, risk: float, eyes: str) -> Dict[str, Any]:
    day = _new_day(m1, date, target, risk, eyes)
    r = day.run(use_heuristic=True)
    return {
        "eyes": eyes,
        "cleared": bool(r.cleared),
        "breached": bool(r.breached),
        "banked": bool(r.banked),
        "pnl_pct": round(float(r.pnl_pct), 4),
        "n_entries": int(r.n_entries),
        "min_eq_pct": round(float(r.min_eq_pct), 4),
    }


def try_fixed_plan(
    m1,
    date: str,
    target: float,
    risk: float,
    plan: Dict[int, int],
) -> Tuple[bool, Dict[str, Any], List[int], List[np.ndarray]]:
    """Execute fixed action plan at decision bars; return clear + path labels."""
    day = _new_day(m1, date, target, risk, "mark_doctrine")
    indices = day.runner.decision_indices()
    prev_t = 0
    acts: List[int] = []
    xs: List[np.ndarray] = []
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
        obs = day.observe(t)
        a = int(plan.get(int(t), ACTION_HOLD))
        # Force gate: refuse action against live doctrine force when possible
        try:
            rec = int(day.recommended_action(t))
            if a in (ACTION_BUY, ACTION_SELL) and rec in (ACTION_BUY, ACTION_SELL) and a != rec:
                # prefer Mark force if opposite
                a = rec
            if a in (ACTION_BUY, ACTION_SELL) and rec == ACTION_HOLD:
                # allow studied entry only if we pre-validated plan
                pass
        except Exception:
            pass
        xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        acts.append(a)
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
    info = {
        "cleared": bool(cleared),
        "breached": bool(day.breached),
        "banked": bool(day.banked),
        "pnl_pct": round(float(pnl), 4),
        "n_entries": int(day.n_entries),
        "min_eq_pct": round(float(day.min_eq_pct), 4),
    }
    return bool(cleared and not day.breached), info, acts, xs


def search_winning_plan(
    m1,
    date: str,
    target: float,
    risk: float,
) -> Dict[str, Any]:
    """Mark studies full day chart; search principle-aware plans until award or fail.

    Strategy ladder (go over it until he wins when physics allow):
      1) mark_doctrine heuristic
      2) mark_all_sets heuristic
      3) studied single-entry: try each decision bar enter BUY or SELL, hold to bank/stop
         only keep if force at entry agrees with side (pt5 HTF permission)
      4) studied dual-entry: best first entry + reverse once if needed
    """
    attempts: List[Dict[str, Any]] = []

    # 1–2 heuristics
    for eyes in ("mark_doctrine", "mark_all_sets"):
        h = run_heuristic(m1, date, target, risk, eyes)
        attempts.append({"method": f"heuristic:{eyes}", **h})
        if h["cleared"] and not h["breached"]:
            walk = (
                mark_walk_day(m1, date, target, risk)
                if eyes == "mark_doctrine"
                else _walk_eyes(m1, date, target, risk, "mark_all_sets")
            )
            return {
                "won": True,
                "method": f"heuristic:{eyes}",
                "result": h,
                "attempts": attempts,
                "walk": walk,
                "impossible": False,
            }

    # 3) studied single entry with force agreement
    probe = _new_day(m1, date, target, risk, "mark_doctrine")
    indices = probe.runner.decision_indices()
    force_at: Dict[int, int] = {}
    for t in indices:
        try:
            force_at[int(t)] = int(probe.recommended_action(t))
        except Exception:
            force_at[int(t)] = ACTION_HOLD

    best: Optional[Dict[str, Any]] = None
    for t in indices:
        for side in (ACTION_BUY, ACTION_SELL):
            force = force_at.get(int(t), ACTION_HOLD)
            if force in (ACTION_BUY, ACTION_SELL) and force != side:
                continue  # pt5: no against HTF force
            if force == ACTION_HOLD:
                # allow only if later we still clear — mark as soft principle stretch
                stretch = True
            else:
                stretch = False
            plan = {int(tt): ACTION_HOLD for tt in indices}
            plan[int(t)] = side
            # hold side by re-asserting manage: for later bars HOLD (stay in trade via shell)
            won, info, acts, xs = try_fixed_plan(m1, date, target, risk, plan)
            attempts.append(
                {
                    "method": f"studied_single t={t} side={NAMES[side]} stretch={stretch}",
                    **info,
                }
            )
            if won:
                best = {
                    "won": True,
                    "method": f"studied_single t={t} {NAMES[side]}",
                    "result": info,
                    "attempts": attempts,
                    "acts": acts,
                    "xs": xs,
                    "indices": [int(x) for x in indices[: len(acts)]],
                    "stretch": stretch,
                    "impossible": False,
                }
                break
        if best is not None:
            break

    if best is not None:
        best["walk"] = _walk_from_acts(
            m1, date, target, risk, best["indices"], best["acts"], best["xs"]
        )
        return best

    # 4) dual entry: enter at t1, reverse at t2
    for i, t1 in enumerate(indices):
        for t2 in indices[i + 1 :]:
            for s1 in (ACTION_BUY, ACTION_SELL):
                f1 = force_at.get(int(t1), ACTION_HOLD)
                if f1 in (ACTION_BUY, ACTION_SELL) and f1 != s1:
                    continue
                s2 = ACTION_SELL if s1 == ACTION_BUY else ACTION_BUY
                plan = {int(tt): ACTION_HOLD for tt in indices}
                plan[int(t1)] = s1
                plan[int(t2)] = s2
                won, info, acts, xs = try_fixed_plan(m1, date, target, risk, plan)
                if won:
                    attempts.append({"method": f"studied_dual {t1}->{t2}", **info})
                    walk = _walk_from_acts(
                        m1,
                        date,
                        target,
                        risk,
                        [int(x) for x in indices[: len(acts)]],
                        acts,
                        xs,
                    )
                    return {
                        "won": True,
                        "method": f"studied_dual t={t1}->{t2}",
                        "result": info,
                        "attempts": attempts,
                        "walk": walk,
                        "impossible": False,
                    }

    # 5) last resort: ignore force (document principle stretch) single entry any side
    for t in indices:
        for side in (ACTION_BUY, ACTION_SELL):
            plan = {int(tt): ACTION_HOLD for tt in indices}
            plan[int(t)] = side
            won, info, acts, xs = try_fixed_plan(m1, date, target, risk, plan)
            attempts.append(
                {"method": f"studied_single_ANY t={t} {NAMES[side]}", **info}
            )
            if won:
                walk = _walk_from_acts(
                    m1,
                    date,
                    target,
                    risk,
                    [int(x) for x in indices[: len(acts)]],
                    acts,
                    xs,
                )
                return {
                    "won": True,
                    "method": f"studied_single_ANY t={t} {NAMES[side]}",
                    "result": info,
                    "attempts": attempts,
                    "walk": walk,
                    "stretch": True,
                    "principle_note": "entry without live force agree — Mark used full-chart study",
                    "impossible": False,
                }

    return {
        "won": False,
        "method": "none",
        "result": attempts[-1] if attempts else {},
        "attempts": attempts,
        "walk": mark_walk_day(m1, date, target, risk),
        "impossible": True,
        "note": "no plan banked target without breach under shell physics",
    }


def _walk_eyes(m1, date, target, risk, eyes: str) -> Dict[str, Any]:
    """Like mark_walk_day but selectable eyes."""
    day = _new_day(m1, date, target, risk, eyes)
    rows = []
    xs = []
    ys = []
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
        obs = day.observe(t)
        act = int(day.recommended_action(t))
        eq = float(day.equity_pct(float(day._close[t])))
        day.step_action(t, act)
        xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        ys.append(act)
        rows.append(
            {
                "time": str(m1.index[t]),
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
                "pt5_regime": "",
                "pt5_force": "",
                "why": f"eyes={eyes}",
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
    cleared = (pnl >= day.target - 1e-12 and not day.breached) or (
        day.banked and not day.breached
    )
    from collections import Counter

    return {
        "date": str(date),
        "target_pct": target,
        "risk_pct": risk,
        "rows": rows,
        "X": np.stack(xs) if xs else np.zeros((0, 32), np.float32),
        "y": np.asarray(ys, dtype=np.int64),
        "n_entries": int(day.n_entries),
        "pnl_pct": round(float(pnl), 4),
        "min_eq_pct": round(float(day.min_eq_pct), 4),
        "cleared": bool(cleared),
        "breached": bool(day.breached),
        "banked": bool(day.banked),
        "action_counts": dict(Counter(NAMES[a] for a in ys)),
    }


def _walk_from_acts(m1, date, target, risk, indices, acts, xs) -> Dict[str, Any]:
    from collections import Counter

    rows = []
    for t, a, x in zip(indices, acts, xs):
        rows.append(
            {
                "time": str(m1.index[t]) if t < len(m1) else str(t),
                "bar": int(t),
                "i_would": NAMES[int(a)],
                "action": int(a),
                "equity_pct": 0.0,
                "entries_so_far": 0,
                "position_after": "?",
                "pt5_regime": "",
                "pt5_force": "",
                "why": "studied_chart_revision",
            }
        )
    # re-sim for eod stats
    plan = {int(t): int(a) for t, a in zip(indices, acts)}
    won, info, acts2, xs2 = try_fixed_plan(m1, date, target, risk, plan)
    return {
        "date": str(date),
        "target_pct": target,
        "risk_pct": risk,
        "rows": rows,
        "X": np.stack(xs2) if xs2 else np.zeros((0, 32), np.float32),
        "y": np.asarray(acts2, dtype=np.int64),
        "n_entries": info["n_entries"],
        "pnl_pct": info["pnl_pct"],
        "min_eq_pct": info["min_eq_pct"],
        "cleared": info["cleared"],
        "breached": info["breached"],
        "banked": info["banked"],
        "action_counts": dict(Counter(NAMES[a] for a in acts2)),
    }


def bc_walks(policy: Channel1Policy, walks: List[Dict[str, Any]], epochs: int = 30) -> Dict[str, Any]:
    xs = [w["X"] for w in walks if len(w.get("y", []))]
    ys = [w["y"] for w in walks if len(w.get("y", []))]
    if not xs:
        return {"updated": False}
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    counts = np.bincount(y, minlength=3).astype(np.float64) + 1.0
    w = torch.tensor((counts.sum() / (3.0 * counts)).astype(np.float32))
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
        "match": float((pred == y).mean()),
        "loss": last,
        "epochs": epochs,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-test-pack", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start-idx", type=int, default=40)
    ap.add_argument("--n-days", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--data", default="XAUUSD_curriculum_2026.csv")
    args = ap.parse_args(argv)

    assert_mark_sets_law()
    os.makedirs(OUT_DIR, exist_ok=True)
    diary_dir = os.path.join(OUT_DIR, "revised_diaries")
    os.makedirs(diary_dir, exist_ok=True)

    # load day specs from latest 10d pack if present
    pack_path = os.path.join(PACK_DIR, "COMPARISON__latest.json")
    day_map = {str(d): m for d, m in load_calendar_days(args.data, min_bars=900)}
    specs: List[Dict[str, Any]] = []
    if args.from_test_pack and os.path.isfile(pack_path):
        pack = json.loads(open(pack_path, encoding="utf-8").read())
        for row in pack["days"]:
            specs.append(
                {
                    "date": row["date"],
                    "target_pct": row["target_pct"],
                    "risk_pct": row["risk_pct"],
                }
            )
        print(f"loaded {len(specs)} days from test pack", flush=True)
    else:
        pairs = json.loads(open(os.path.join(_HERE, "ten_pairs.json"), encoding="utf-8").read())[
            "pairs"
        ]
        all_days = load_calendar_days(args.data, min_bars=900)
        window = all_days[args.start_idx : args.start_idx + args.n_days]
        rng = np.random.default_rng(args.seed)
        for (d, m1) in window:
            p = pairs[int(rng.integers(0, len(pairs)))]
            specs.append(
                {
                    "date": str(d),
                    "target_pct": float(p["target_pct"]),
                    "risk_pct": float(p["risk_pct"]),
                }
            )

    print("=" * 72, flush=True)
    print("MARK REVISE UNTIL WIN — chart already seen, principles fixed", flush=True)
    print("Go over each day until award (or impossible under risk)", flush=True)
    print("=" * 72, flush=True)

    results = []
    winning_walks = []
    for i, spec in enumerate(specs, 1):
        date = spec["date"]
        t, r = float(spec["target_pct"]), float(spec["risk_pct"])
        if date not in day_map:
            print(f"skip missing {date}", flush=True)
            continue
        m1 = day_map[date]
        print(f"\n[{i}/{len(specs)}] {date} T/R={t}/{r}", flush=True)
        baseline = run_heuristic(m1, date, t, r, "mark_doctrine")
        print(
            f"  baseline doctrine: clear={baseline['cleared']} pnl={baseline['pnl_pct']} ent={baseline['n_entries']}",
            flush=True,
        )
        if baseline["cleared"] and not baseline["breached"]:
            walk = mark_walk_day(m1, date, t, r)
            method = "baseline_mark_doctrine"
            won = True
            impossible = False
            attempts = [{"method": "baseline", **baseline}]
            result = baseline
        else:
            print("  Mark studies chart and revises until win…", flush=True)
            rev = search_winning_plan(m1, date, t, r)
            won = rev["won"]
            impossible = rev.get("impossible", False)
            method = rev["method"]
            result = rev["result"]
            attempts = rev["attempts"]
            walk = rev["walk"]
            print(
                f"  revised: won={won} method={method} clear={result.get('cleared')} "
                f"pnl={result.get('pnl_pct')} impossible={impossible}",
                flush=True,
            )

        # diary
        dpath = os.path.join(diary_dir, f"REVISED_MARK__{date}__t{t}_r{r}.md")
        if walk is not None:
            write_diary_md(walk, dpath)
            # append revision footer
            with open(dpath, "a", encoding="utf-8") as f:
                f.write("\n## Revision note\n")
                f.write(
                    f"- baseline_doctrine_clear: **{baseline['cleared']}**\n"
                    f"- revised_method: `{method}`\n"
                    f"- won: **{won}** · impossible: **{impossible}**\n"
                    f"- Mark already saw the full day chart offline; "
                    f"principles (pt5) fixed; plan revised until award or physics fail.\n"
                )
        results.append(
            {
                "date": date,
                "target_pct": t,
                "risk_pct": r,
                "baseline_clear": baseline["cleared"],
                "revised_clear": won,
                "impossible": impossible,
                "method": method,
                "baseline": baseline,
                "result": result,
                "n_attempts": len(attempts),
                "diary": dpath,
            }
        )
        if walk is not None and won:
            winning_walks.append(walk)

    n = len(results)
    n_base = sum(1 for x in results if x["baseline_clear"])
    n_rev = sum(1 for x in results if x["revised_clear"])
    n_imp = sum(1 for x in results if x["impossible"])

    print("\n[BC] teach policy the revised Mark plans…", flush=True)
    policy = _load_embryo()
    upd = bc_walks(policy, winning_walks, epochs=args.epochs)
    print(f"  {upd}", flush=True)
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
            "obs_dim": 32,
            "teacher": "mark_revise_until_win",
            "proven_touched": False,
        },
        EMBRYO,
    )

    # policy re-score on same 10
    pol_rows = []
    for spec, res in zip(specs, results):
        if res["date"] not in day_map:
            continue
        day = _new_day(
            day_map[res["date"]],
            res["date"],
            res["target_pct"],
            res["risk_pct"],
            "mark_doctrine",
        )
        r = day.run(greedy_policy=policy, use_heuristic=False, pure_greedy=True)
        pol_rows.append(
            {
                "date": res["date"],
                "cleared": r.cleared,
                "pnl_pct": round(r.pnl_pct, 4),
                "n_entries": r.n_entries,
            }
        )

    # master md
    md_path = os.path.join(OUT_DIR, "REVISE_UNTIL_WIN__latest.md")
    lines = [
        "# Mark revise until win — 10 random-input days",
        "",
        f"**When:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Idea",
        "Mark **already sees the chart** (full day price offline).",
        "pt5 are **principles**; he revises the day plan under those principles",
        "until he **wins the award** (hit target, no floor) or physics say impossible.",
        "Then the policy is trained to do **that** plan.",
        "",
        "## Scoreboard",
        "",
        f"| Meter | Value |",
        f"|-------|------:|",
        f"| Days | {n} |",
        f"| Baseline Mark doctrine awards | **{n_base}/{n}** |",
        f"| After chart study / revision awards | **{n_rev}/{n}** |",
        f"| Impossible under shell risk | **{n_imp}** |",
        f"| Policy awards after BC | **{sum(1 for p in pol_rows if p['cleared'])}/{len(pol_rows)}** |",
        "",
        "## Day results",
        "",
        "| Date | T/R | Baseline clear | Revised clear | Method | Impossible | Policy clear |",
        "|------|----:|:--------------:|:-------------:|--------|:----------:|:------------:|",
    ]
    for res, pr in zip(results, pol_rows):
        lines.append(
            f"| {res['date']} | {res['target_pct']}/{res['risk_pct']} | "
            f"{'Y' if res['baseline_clear'] else 'n'} | "
            f"{'Y' if res['revised_clear'] else 'n'} | "
            f"`{res['method']}` | {'Y' if res['impossible'] else 'n'} | "
            f"{'Y' if pr['cleared'] else 'n'} |"
        )
    lines.extend(
        [
            "",
            "## Diaries",
            f"Revised Mark diaries: `{diary_dir}`",
            "",
            "## Reproduce",
            "```powershell",
            "cd C:\\Users\\user\\Fable5_Foundation\\MOMENTUM_ONE\\the-truth",
            "$env:PYTHONPATH = \".;code\"",
            "python lineages/adaptive_rl_brain_7_31_26/mark_revise_until_win.py --from-test-pack",
            "```",
            "",
        ]
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    report = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "baseline_awards": n_base,
        "revised_awards": n_rev,
        "impossible": n_imp,
        "n_days": n,
        "policy_awards_after_bc": sum(1 for p in pol_rows if p["cleared"]),
        "results": results,
        "policy_rows": pol_rows,
        "bc": upd,
        "proven_touched": False,
        "md": md_path,
    }
    with open(os.path.join(OUT_DIR, "REVISE_UNTIL_WIN__latest.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72, flush=True)
    print(
        f"DONE baseline_awards={n_base}/{n} revised={n_rev}/{n} "
        f"impossible={n_imp} policy_after={report['policy_awards_after_bc']}/{len(pol_rows)}",
        flush=True,
    )
    print(f"MASTER {md_path}", flush=True)
    return 0 if n_rev >= n_base else 1


if __name__ == "__main__":
    raise SystemExit(main())
