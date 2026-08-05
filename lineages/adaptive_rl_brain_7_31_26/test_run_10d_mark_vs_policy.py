"""TEST RUN: 10 days · random target/risk · Mark (pt5) first · then policy.

Protocol:
  1. Sample 10 chronological days + random (target%, risk%) per day (same seed = reproducible)
  2. MARK first: doctrine + MARK SETS LAW + pt5 principles → day diary + labels
  3. POLICY second: pure greedy embryo on same days/pairs
  4. Write one master comparison pack (markdown + json)

Usage (repo root, PYTHONPATH=.;code):
  python lineages/adaptive_rl_brain_7_31_26/test_run_10d_mark_vs_policy.py
  python lineages/adaptive_rl_brain_7_31_26/test_run_10d_mark_vs_policy.py --seed 7 --start-idx 40
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay, load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.mark_day_diary import (
    PT5_PRINCIPLES,
    bc_to_mark,
    mark_walk_day,
    policy_compare,
    write_diary_md,
    _load_embryo,
)
from lineages.adaptive_rl_brain_7_31_26.perception.sets import (
    assert_mark_sets_law,
    mark_sets_law_table,
)
from lineages.adaptive_rl_brain_7_31_26.policy_stub import Channel1Policy

CKPT_DIR = os.path.join(_HERE, "checkpoints")
OUT_DIR = os.path.join(CKPT_DIR, "test_run_10d_mark_vs_policy")
PAIRS_PATH = os.path.join(_HERE, "ten_pairs.json")
NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}


def load_pairs() -> List[Tuple[float, float]]:
    with open(PAIRS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)["pairs"]
    return [(float(p["target_pct"]), float(p["risk_pct"])) for p in raw]


def run_policy_day(
    m1,
    date_str: str,
    target: float,
    risk: float,
    policy: Channel1Policy,
    *,
    full_obs: bool = False,
) -> Dict[str, Any]:
    day = GoalEquityDay(
        m1,
        target_pct=target,
        risk_pct=risk,
        date_str=str(date_str),
        eyes_mode="mark_doctrine",
        mark_soul=True,
        full_obs=full_obs,
    )
    r = day.run(greedy_policy=policy, use_heuristic=False, pure_greedy=True)
    return {
        "date": str(date_str),
        "target_pct": target,
        "risk_pct": risk,
        "cleared": bool(r.cleared),
        "breached": bool(r.breached),
        "banked": bool(r.banked),
        "pnl_pct": round(float(r.pnl_pct), 4),
        "min_eq_pct": round(float(r.min_eq_pct), 4),
        "n_entries": int(r.n_entries),
        "hold_rate": round(float(r.hold_rate), 4),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="10-day Mark diary vs policy test pack")
    ap.add_argument("--n-days", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--start-idx",
        type=int,
        default=40,
        help="start index into calendar days (>=900 bars)",
    )
    ap.add_argument("--data", default="XAUUSD_curriculum_2026.csv")
    ap.add_argument(
        "--also-bc",
        action="store_true",
        help="after document, BC policy on these 10 days and re-compare",
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument(
        "--full-obs",
        action="store_true",
        help="policy uses 168-dim Mark full board (sets+doctrine+92 agents+self)",
    )
    args = ap.parse_args(argv)

    assert_mark_sets_law()
    os.makedirs(OUT_DIR, exist_ok=True)
    diary_dir = os.path.join(OUT_DIR, "mark_diaries")
    os.makedirs(diary_dir, exist_ok=True)

    all_days = load_calendar_days(args.data, min_bars=900)
    pairs = load_pairs()
    rng = np.random.default_rng(args.seed)

    start = max(0, int(args.start_idx))
    end = min(len(all_days), start + int(args.n_days))
    window = all_days[start:end]
    if len(window) < int(args.n_days):
        # wrap / take last n
        window = all_days[-int(args.n_days) :]

    day_specs: List[Dict[str, Any]] = []
    for (date_str, m1) in window:
        t, r = pairs[int(rng.integers(0, len(pairs)))]
        day_specs.append(
            {
                "date": str(date_str),
                "target_pct": t,
                "risk_pct": r,
                "m1": m1,
            }
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print("=" * 72, flush=True)
    print("TEST RUN — 10 days random inputs", flush=True)
    print("Phase A: MARK (pt5 principles) documents each day first", flush=True)
    print("Phase B: POLICY runs the same days", flush=True)
    print(f"seed={args.seed} start_idx={start} n={len(day_specs)}", flush=True)
    print("=" * 72, flush=True)

    # --- Phase A: Mark first ---
    mark_walks: List[Dict[str, Any]] = []
    for i, spec in enumerate(day_specs, 1):
        print(
            f"\n[A {i}/{len(day_specs)}] MARK chart-first  {spec['date']}  "
            f"T/R={spec['target_pct']}/{spec['risk_pct']}",
            flush=True,
        )
        w = mark_walk_day(
            spec["m1"],
            spec["date"],
            spec["target_pct"],
            spec["risk_pct"],
            full_obs=bool(args.full_obs),
        )
        dpath = os.path.join(
            diary_dir,
            f"MARK_DIARY__{spec['date']}__t{spec['target_pct']}_r{spec['risk_pct']}.md",
        )
        write_diary_md(w, dpath)
        print(
            f"    diary → {os.path.basename(dpath)}  "
            f"entries={w['n_entries']} clear={w['cleared']} pnl={w['pnl_pct']} "
            f"actions={w['action_counts']}",
            flush=True,
        )
        # strip heavy arrays for summary later
        mark_walks.append(w)

    # --- Phase B: Policy second ---
    print("\n" + "=" * 72, flush=True)
    print("Phase B: POLICY pure greedy (same days + same random T/R)", flush=True)
    print("=" * 72, flush=True)
    policy = _load_embryo(full_obs=bool(args.full_obs))
    policy_rows: List[Dict[str, Any]] = []
    bar_compares: List[Dict[str, Any]] = []
    print(
        f"policy obs_dim={getattr(policy, 'obs_dim', '?')} full_obs={args.full_obs}",
        flush=True,
    )

    for i, (spec, w) in enumerate(zip(day_specs, mark_walks), 1):
        print(
            f"\n[B {i}/{len(day_specs)}] POLICY  {spec['date']}  "
            f"T/R={spec['target_pct']}/{spec['risk_pct']}",
            flush=True,
        )
        pr = run_policy_day(
            spec["m1"],
            spec["date"],
            spec["target_pct"],
            spec["risk_pct"],
            policy,
            full_obs=bool(args.full_obs),
        )
        bc = policy_compare(w, policy)
        print(
            f"    policy entries={pr['n_entries']} clear={pr['cleared']} pnl={pr['pnl_pct']}  "
            f"| bar-agree with Mark diary={bc['agree_rate']:.3f} "
            f"({bc['agree']}/{bc['n']})",
            flush=True,
        )
        policy_rows.append(pr)
        bar_compares.append({"date": spec["date"], **bc})

    # optional BC + recompare
    post = None
    if args.also_bc:
        print("\n[C] BC policy on these Mark diaries…", flush=True)
        upd = bc_to_mark(policy, mark_walks, epochs=args.epochs)
        print(f"    {upd}", flush=True)
        post = []
        for w in mark_walks:
            post.append({"date": w["date"], **policy_compare(w, policy)})
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
                "teacher": "test_run_10d_mark_diaries",
                "proven_touched": False,
            },
            os.path.join(CKPT_DIR, "mark_clone_doctrine_v1.pt"),
        )

    # --- Master report ---
    comparison_rows = []
    for spec, w, pr, bc in zip(day_specs, mark_walks, policy_rows, bar_compares):
        comparison_rows.append(
            {
                "date": spec["date"],
                "target_pct": spec["target_pct"],
                "risk_pct": spec["risk_pct"],
                "mark_cleared": w["cleared"],
                "policy_cleared": pr["cleared"],
                "mark_breached": w["breached"],
                "policy_breached": pr["breached"],
                "mark_entries": w["n_entries"],
                "policy_entries": pr["n_entries"],
                "mark_pnl": w["pnl_pct"],
                "policy_pnl": pr["pnl_pct"],
                "mark_actions": w["action_counts"],
                "bar_agree_rate": bc["agree_rate"],
                "bar_disagree": bc["disagree"],
                "same_clear_outcome": w["cleared"] == pr["cleared"],
                "diary": f"mark_diaries/MARK_DIARY__{spec['date']}__t{spec['target_pct']}_r{spec['risk_pct']}.md",
            }
        )

    n = len(comparison_rows)
    mean_agree = float(np.mean([r["bar_agree_rate"] for r in comparison_rows])) if n else 0.0
    mark_awards = sum(1 for r in comparison_rows if r["mark_cleared"])
    pol_awards = sum(1 for r in comparison_rows if r["policy_cleared"])
    same_outcome = sum(1 for r in comparison_rows if r["same_clear_outcome"])
    breaches = sum(
        1 for r in comparison_rows if r["mark_breached"] or r["policy_breached"]
    )

    # consecutive award streaks within the 10
    def max_streak(key: str) -> int:
        best = cur = 0
        for r in comparison_rows:
            if r[key]:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    pack = {
        "title": "TEST RUN 10 days — Mark (pt5) first vs policy",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "stamp": stamp,
        "protocol": [
            "1. Random (target,risk) from ten_pairs.json per day",
            "2. MARK documents exact day under pt5 principles + sets law",
            "3. POLICY pure greedy on same days/pairs",
            "4. Compare bar-agree and award/clear outcomes",
        ],
        "pt5_note": "pt5 = principles of how Mark trades (not full discretionary system)",
        "pt5_principles": PT5_PRINCIPLES,
        "sets_law": mark_sets_law_table(),
        "seed": args.seed,
        "start_idx": start,
        "n_days": n,
        "pairs_source": "ten_pairs.json",
        "unique_pairs_used": sorted(
            {(r["target_pct"], r["risk_pct"]) for r in comparison_rows}
        ),
        "summary": {
            "mean_bar_agree_mark_vs_policy": mean_agree,
            "mark_award_days": mark_awards,
            "policy_award_days": pol_awards,
            "same_clear_outcome_days": same_outcome,
            "breaches_either": breaches,
            "mark_max_clear_streak_in_window": max_streak("mark_cleared"),
            "policy_max_clear_streak_in_window": max_streak("policy_cleared"),
            "random_inputs_no_retrain": True,
            "proven_touched": False,
        },
        "days": comparison_rows,
        "post_bc": post,
        "out_dir": OUT_DIR,
    }

    json_path = os.path.join(OUT_DIR, f"COMPARISON__{stamp}.json")
    latest_json = os.path.join(OUT_DIR, "COMPARISON__latest.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pack, f, indent=2)
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(pack, f, indent=2)

    # Master markdown
    md_path = os.path.join(OUT_DIR, f"TEST_RUN_10D__{stamp}.md")
    latest_md = os.path.join(OUT_DIR, "TEST_RUN_10D__latest.md")
    lines: List[str] = []
    lines.append("# TEST RUN — 10 days · random inputs · Mark first · then policy")
    lines.append("")
    lines.append(f"**When:** {pack['saved_at']}")
    lines.append(f"**Seed:** `{args.seed}` · **start_idx:** {start} · **n_days:** {n}")
    lines.append(
        f"**Random pairs used:** `{pack['unique_pairs_used']}` "
        f"(from ten_pairs.json — **no retrain** between days)"
    )
    lines.append("")
    lines.append("## Protocol")
    for i, p in enumerate(pack["protocol"], 1):
        step = p.split(". ", 1)[-1] if ". " in p[:4] else p
        lines.append(f"{i}. {step}")
    lines.append("")
    lines.append("## What “Mark” means in this test")
    lines.append("")
    lines.append(
        "- **pt5** (`all llm's have to know… pt5`) = **principles** of how Mark trades "
        "(HTF permission, LTF timing, breath/launch, regime, capital) — not a full human discretionary log."
    )
    lines.append(
        "- **MARK SETS LAW** = four stacks (LTF first for pullback/cont/add; last two HTF confirm)."
    )
    lines.append(
        "- **Mark actions** = codified teacher (`eyes_mode=mark_doctrine`) applied to **real day price**."
    )
    lines.append(
        "- **Policy** = pure greedy `mark_clone_doctrine_v1.pt` on the **same** obs/shell/pairs."
    )
    lines.append(
        "- `01_SYSTEM/config/models` is **which LLM** runs Army work (cost) — **not** used for trade side here."
    )
    lines.append("")
    lines.append("### pt5 principles cited")
    for p in PT5_PRINCIPLES:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("### Sets scanned every decision")
    for row in mark_sets_law_table():
        lines.append(
            f"- Set {row['set_id']}: **{row['ltf_entry']}** | "
            f"{', '.join(row['htf_confirm'])}"
        )
    lines.append("")
    lines.append("## Summary scoreboard")
    lines.append("")
    lines.append("| Meter | Mark (pt5 teacher) | Policy |")
    lines.append("|-------|-------------------:|-------:|")
    lines.append(f"| Award/clear days / {n} | **{mark_awards}** | **{pol_awards}** |")
    lines.append(
        f"| Max clear streak in window | **{max_streak('mark_cleared')}** | "
        f"**{max_streak('policy_cleared')}** |"
    )
    lines.append(f"| Breaches | see days | see days |")
    lines.append(f"| Mean bar-agree (policy vs Mark diary) | — | **{mean_agree:.1%}** |")
    lines.append(f"| Days with same clear outcome | **{same_outcome}/{n}** | |")
    lines.append(f"| Breach events (either) | **{breaches}** | |")
    lines.append("")
    lines.append("## Day-by-day comparison")
    lines.append("")
    lines.append(
        "| Date | T/R | Mark clear | Pol clear | Mark ent | Pol ent | Mark pnl | Pol pnl | Bar agree | Diary |"
    )
    lines.append(
        "|------|----:|:----------:|:---------:|---------:|--------:|---------:|--------:|----------:|-------|"
    )
    for r in comparison_rows:
        lines.append(
            f"| {r['date']} | {r['target_pct']}/{r['risk_pct']} | "
            f"{'Y' if r['mark_cleared'] else 'n'} | "
            f"{'Y' if r['policy_cleared'] else 'n'} | "
            f"{r['mark_entries']} | {r['policy_entries']} | "
            f"{r['mark_pnl']} | {r['policy_pnl']} | "
            f"{r['bar_agree_rate']:.0%} | `{r['diary']}` |"
        )
    lines.append("")
    lines.append("## What this tells us")
    lines.append("")
    if mean_agree >= 0.85 and mark_awards == pol_awards:
        lines.append(
            "- **Close:** policy mostly does what Mark’s written day does on these 10."
        )
    elif mean_agree >= 0.7:
        lines.append(
            "- **Partial:** policy often matches side/timing on bars, but day outcomes still diverge."
        )
    else:
        lines.append(
            "- **Gap:** policy does **not** yet do the same thing Mark wrote for the day."
        )
    lines.append(
        f"- Mark awards **{mark_awards}/{n}**; policy awards **{pol_awards}/{n}** "
        f"under the **same random inputs**."
    )
    lines.append(
        "- Read each `mark_diaries/MARK_DIARY__*.md` for exact bar-by-bar “I would …” under pt5."
    )
    lines.append("")
    lines.append("## Files")
    lines.append(f"- Master MD: `{md_path}`")
    lines.append(f"- Master JSON: `{json_path}`")
    lines.append(f"- Diaries dir: `{diary_dir}`")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("```powershell")
    lines.append("cd C:\\Users\\user\\Fable5_Foundation\\MOMENTUM_ONE\\the-truth")
    lines.append("$env:PYTHONPATH = \".;code\"")
    lines.append(
        f"python lineages/adaptive_rl_brain_7_31_26/test_run_10d_mark_vs_policy.py "
        f"--seed {args.seed} --start-idx {start} --n-days {args.n_days}"
    )
    lines.append("```")
    lines.append("")

    md_text = "\n".join(lines)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    with open(latest_md, "w", encoding="utf-8") as f:
        f.write(md_text)

    # INDEX
    index_path = os.path.join(OUT_DIR, "00_READ_ME_FIRST.md")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    "# 10-day Mark vs policy test pack",
                    "",
                    f"**Latest report:** [TEST_RUN_10D__latest.md](TEST_RUN_10D__latest.md)",
                    f"**Latest JSON:** [COMPARISON__latest.json](COMPARISON__latest.json)",
                    "",
                    "## Order of reading",
                    "1. This file",
                    "2. `TEST_RUN_10D__latest.md` (scoreboard)",
                    "3. Each `mark_diaries/MARK_DIARY__*.md` (what Mark would do that day)",
                    "4. JSON for machine metrics",
                    "",
                    "## Protocol reminder",
                    "Mark (pt5 principles + sets) documents the day **first**.",
                    "Policy runs **second** on the same random target/risk.",
                    "Then we know the gap.",
                    "",
                ]
            )
        )

    print("\n" + "=" * 72, flush=True)
    print(
        f"DONE mean_bar_agree={mean_agree:.3f} mark_awards={mark_awards}/{n} "
        f"policy_awards={pol_awards}/{n} same_outcome={same_outcome}/{n} breach={breaches}",
        flush=True,
    )
    print(f"MASTER {latest_md}", flush=True)
    print(f"JSON   {latest_json}", flush=True)
    print(f"DIARIES {diary_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
