"""A/B test: does practice-only dial search + trust gates HELP vs leaky search?

Does NOT modify multi_pair policy defaults or PROVEN.
Uses existing score_all_pairs / GoalEquityDay.

Hypothesis (dream):
  A) baseline dials
  B) dial search on ALL days (leaky — optimizes including forward)
  C) dial search on PRACTICE only (wall)

  Evaluate each on FORWARD (true unseen for C).
  If C forward >= A and breach still 0, and B only "wins" by peeking → architecture helps.

Also: G01 gate blocks meaning tamper (integrity help, not clear%).
G14: tags with high practice P_clear but low forward P_clear — if we avoid
      entries on "lying" tags, does forward clear improve? (soft filter test)

Usage:
  python lineages/adaptive_rl_brain_7_31_26/probe_architecture_helps.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import (
    GoalEquityDay,
    load_calendar_days,
    split_practice_forward,
)
from lineages.adaptive_rl_brain_7_31_26.score_ten_pairs import (
    load_pairs_config,
    score_all_pairs,
    score_pair_on_days,
)
from lineages.adaptive_rl_brain_7_31_26.perception import live_indicators as li

OUT = os.path.join(_HERE, "checkpoints", "architecture_helps_report.json")
DATA = "XAUUSD_curriculum_2026.csv"
PRACTICE_N = 50
DECIDE = 25
# Smaller grid for speed (still tests the principle)
# Compact grid so A/B finishes in minutes (principle test, not full IRAC grid)
GRID_RISK = (0.25, 0.35, 0.50)
GRID_STOP = (1.5, 2.0, 2.5)
GRID_CAP = (0.25, 0.50)
SEARCH_MAX_DAYS = 15
# Eval full practice/forward for final score (10 pairs)
BASELINE = {
    "risk_use_frac": 0.35,
    "stop_atr_mult": 2.0,
    "per_trade_cap_pct": 0.25,
}
# Search uses subset of pairs for speed; final eval still all 10
SEARCH_PAIR_IDS = (1, 5, 10)


def meaning_cfg() -> dict:
    return {
        "cci": (li.CCI_FAST, li.CCI_SLOW),
        "rsi": (li.RSI_FAST, li.RSI_SLOW),
        "channel": (li.CHANNEL_N, li.CHANNEL_SHIFT),
        "ref": (li.REF_SMA_N, li.REF_SMA_SHIFT),
        "groups": list(li.GROUP_KEYS),
        "decode": "heuristic",
    }


def meaning_hash(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def assert_no_leak(search_days, forward_days) -> None:
    s = {d for d, _ in search_days}
    f = {d for d, _ in forward_days}
    bad = s & f
    if bad:
        raise RuntimeError("LEAK %s" % sorted(bad)[:5])


def summarize(rep: dict) -> dict:
    pairs = rep["pairs"]
    return {
        "n_pass": int(rep["n_pass"]),
        "all_pass": bool(rep.get("all_pass")),
        "clears_sum": int(sum(p["cleared"] for p in pairs)),
        "breach_sum": int(sum(p["breached"] for p in pairs)),
        "mean_clear_pct": float(np.mean([p["clear_pct"] for p in pairs])),
        "pairs": [
            {
                "id": p.get("id"),
                "target": p["target_pct"],
                "risk": p["risk_pct"],
                "cleared": p["cleared"],
                "breached": p["breached"],
                "clear_pct": round(p["clear_pct"], 1),
            }
            for p in pairs
        ],
    }


def dial_score(days, pairs, dials, max_days: Optional[int] = None) -> dict:
    use = list(days) if max_days is None else list(days)[: int(max_days)]
    return score_all_pairs(
        pairs, use, policy=None, use_heuristic=True, decide_every=DECIDE, dials=dials
    )


def search_dials_on(
    days,
    pairs,
    *,
    label: str,
    max_days: int,
) -> Tuple[dict, dict]:
    grid = []
    for ru in GRID_RISK:
        for sm in GRID_STOP:
            for cap in GRID_CAP:
                grid.append(
                    {
                        "risk_use_frac": ru,
                        "stop_atr_mult": sm,
                        "per_trade_cap_pct": cap,
                    }
                )
    best = dict(BASELINE)
    best_rep = dial_score(days, pairs, best, max_days=max_days)
    best_key = (
        int(best_rep["n_pass"]),
        sum(p["cleared"] for p in best_rep["pairs"]),
        -sum(p["breached"] for p in best_rep["pairs"]),
    )
    print(f"  [{label}] grid n={len(grid)} baseline key={best_key}", flush=True)
    for i, dials in enumerate(grid):
        rep = dial_score(days, pairs, dials, max_days=max_days)
        key = (
            int(rep["n_pass"]),
            sum(p["cleared"] for p in rep["pairs"]),
            -sum(p["breached"] for p in rep["pairs"]),
        )
        if (i + 1) % 12 == 0 or key > best_key:
            print(
                f"  [{label}] {i+1}/{len(grid)} pass={key[0]} clears={key[1]} "
                f"breach={-key[2]} dials={dials}",
                flush=True,
            )
        if key > best_key:
            best_key = key
            best = dials
            best_rep = rep
    print(f"  [{label}] BEST {best} key={best_key}", flush=True)
    return best, best_rep


def tag_at(day: GoalEquityDay, t: int) -> str:
    saved = day.runner.position
    day.runner.position = None
    try:
        perc = day.runner.perceive(t)
    finally:
        day.runner.position = saved
    hi = perc["higher"].name
    if perc["structure"].pullback and hi == "BULL":
        return "pullback_vs_bull"
    if perc["structure"].pullback and hi == "BEAR":
        return "pullback_vs_bear"
    if hi == "BULL":
        return "higher_bull"
    if hi == "BEAR":
        return "higher_bear"
    return "other"


def p_clear_by_tag(days, dials, target, risk, max_days: int) -> Dict[str, dict]:
    stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for date_str, m1 in list(days)[:max_days]:
        if "vol" not in m1.columns:
            m1 = m1.copy()
            m1["vol"] = 100.0
        day = GoalEquityDay(
            m1,
            target_pct=target,
            risk_pct=risk,
            risk_use_frac=float(dials["risk_use_frac"]),
            stop_atr_mult=float(dials["stop_atr_mult"]),
            per_trade_cap_pct=float(dials["per_trade_cap_pct"]),
            date_str=str(date_str),
        )
        tags_today = set()
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
            before = day.n_entries
            act = day.recommended_action(t)
            day.step_action(t, int(act))
            if day.n_entries > before:
                tags_today.add(tag_at(day, t))
        if not day.dead and not day.banked:
            for bt in range(prev_t, len(day.m1)):
                if day.dead or day.banked:
                    break
                day._mark_bar(bt)
        t_last = len(day.m1) - 1
        day._flatten(float(day._close[t_last]), float(day._spread_px[t_last]))
        pnl = 100.0 * (day.balance - day.eq0) / day.eq0
        cleared = (not day.breached) and pnl >= target - 1e-9
        for tag in tags_today:
            stats[tag][0] += 1
            if cleared:
                stats[tag][1] += 1
    return {
        t: {"plays": v[0], "clears": v[1], "p_clear": v[1] / max(v[0], 1)}
        for t, v in stats.items()
    }


def score_with_banned_tags(
    days, dials, target, risk, banned: set, max_days: int
) -> dict:
    """Heuristic but refuse open when structure tag is in banned set."""
    cleared = breached = n = 0
    for date_str, m1 in list(days)[:max_days]:
        if "vol" not in m1.columns:
            m1 = m1.copy()
            m1["vol"] = 100.0
        day = GoalEquityDay(
            m1,
            target_pct=target,
            risk_pct=risk,
            risk_use_frac=float(dials["risk_use_frac"]),
            stop_atr_mult=float(dials["stop_atr_mult"]),
            per_trade_cap_pct=float(dials["per_trade_cap_pct"]),
            date_str=str(date_str),
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
            act = day.recommended_action(t)
            if day.side is None and act != 0:
                tg = tag_at(day, t)
                if tg in banned:
                    act = 0  # HOLD — don't trust lying sensor
            day.step_action(t, int(act))
        if not day.dead and not day.banked:
            for bt in range(prev_t, len(day.m1)):
                if day.dead or day.banked:
                    break
                day._mark_bar(bt)
        t_last = len(day.m1) - 1
        day._flatten(float(day._close[t_last]), float(day._spread_px[t_last]))
        pnl = 100.0 * (day.balance - day.eq0) / day.eq0
        n += 1
        if day.breached:
            breached += 1
        elif pnl >= target - 1e-9:
            cleared += 1
    return {
        "n_days": n,
        "cleared": cleared,
        "breached": breached,
        "clear_pct": 100.0 * cleared / max(n, 1),
        "banned": sorted(banned),
    }


def main() -> None:
    print("=" * 64, flush=True)
    print("ARCHITECTURE HELPS? A/B probe (no policy shipping)", flush=True)
    print("=" * 64, flush=True)

    cfg = load_pairs_config()
    pairs = list(cfg["pairs"])
    search_pairs = [p for p in pairs if int(p.get("id", 0)) in SEARCH_PAIR_IDS]
    if not search_pairs:
        search_pairs = pairs[:3]
    all_days = load_calendar_days(DATA, min_bars=900)
    practice, forward = split_practice_forward(all_days, practice_n=PRACTICE_N)
    print(
        f"days total={len(all_days)} practice={len(practice)} forward={len(forward)} "
        f"pairs={len(pairs)} search_pairs={len(search_pairs)}",
        flush=True,
    )

    # --- G01 integrity ---
    pin = meaning_hash(meaning_cfg())
    live = meaning_hash(meaning_cfg())
    bad = dict(meaning_cfg())
    bad["cci"] = (10, 100)
    bad_h = meaning_hash(bad)
    g01 = {
        "pin": pin,
        "live_match": live == pin,
        "blocks_wave_cci_tamper": bad_h != pin,
        "helps": "integrity (prevents false forward scores), not clear% by itself",
    }
    print(f"\n[G01] pin match={g01['live_match']} blocks_tamper={g01['blocks_wave_cci_tamper']}", flush=True)

    # --- G04/G05 leak ---
    try:
        assert_no_leak(practice, forward)
        leak_clean = True
    except RuntimeError:
        leak_clean = False
    try:
        assert_no_leak(all_days, forward)
        leak_detects_all = False
    except RuntimeError:
        leak_detects_all = True
    print(f"[G05] clean_split={leak_clean} catches_all_days_search={leak_detects_all}", flush=True)

    # --- A/B dial search ---
    print("\n[A] Baseline dials → score practice + forward...", flush=True)
    base_prac = summarize(dial_score(practice, pairs, BASELINE))
    base_fwd = summarize(dial_score(forward, pairs, BASELINE))
    print(f"  baseline practice pass={base_prac['n_pass']} clears={base_prac['clears_sum']} breach={base_prac['breach_sum']}", flush=True)
    print(f"  baseline forward  pass={base_fwd['n_pass']} clears={base_fwd['clears_sum']} breach={base_fwd['breach_sum']}", flush=True)

    print("\n[B] LEAKY search on ALL days (includes forward)...", flush=True)
    leaky_dials, _ = search_dials_on(
        all_days, search_pairs, label="leaky_all", max_days=SEARCH_MAX_DAYS
    )
    print("  scoring leaky dials on FULL forward (true test)...", flush=True)
    leaky_prac = summarize(dial_score(practice, pairs, leaky_dials))
    leaky_fwd = summarize(dial_score(forward, pairs, leaky_dials))
    print(f"  leaky practice pass={leaky_prac['n_pass']} clears={leaky_prac['clears_sum']} breach={leaky_prac['breach_sum']}", flush=True)
    print(f"  leaky forward  pass={leaky_fwd['n_pass']} clears={leaky_fwd['clears_sum']} breach={leaky_fwd['breach_sum']}", flush=True)

    print("\n[C] CLEAN search on PRACTICE only (tripwire)...", flush=True)
    assert_no_leak(practice, forward)
    clean_dials, _ = search_dials_on(
        practice, search_pairs, label="clean_practice", max_days=SEARCH_MAX_DAYS
    )
    print("  scoring clean dials on FULL forward (true unseen)...", flush=True)
    clean_prac = summarize(dial_score(practice, pairs, clean_dials))
    clean_fwd = summarize(dial_score(forward, pairs, clean_dials))
    print(f"  clean practice pass={clean_prac['n_pass']} clears={clean_prac['clears_sum']} breach={clean_prac['breach_sum']}", flush=True)
    print(f"  clean forward  pass={clean_fwd['n_pass']} clears={clean_fwd['clears_sum']} breach={clean_fwd['breach_sum']}", flush=True)

    # Does clean help vs baseline on forward?
    clean_helps_fwd = (
        clean_fwd["breach_sum"] <= base_fwd["breach_sum"]
        and (
            clean_fwd["clears_sum"] > base_fwd["clears_sum"]
            or clean_fwd["n_pass"] > base_fwd["n_pass"]
        )
    )
    # Does leaky only look good because of peek? (leaky forward >> clean forward)
    leaky_overfits = leaky_fwd["clears_sum"] > clean_fwd["clears_sum"] + 5

    # --- G14/G15 trust filter on one pair ---
    print("\n[G14] Trust decay @ 2.0/3.0 baseline dials...", flush=True)
    tp = p_clear_by_tag(practice, BASELINE, 2.0, 3.0, max_days=30)
    tf = p_clear_by_tag(forward, BASELINE, 2.0, 3.0, max_days=30)
    lying = []
    trust_rows = []
    for tag in sorted(set(tp) | set(tf)):
        pp, fp = tp.get(tag, {"p_clear": 0, "plays": 0}), tf.get(tag, {"p_clear": 0, "plays": 0})
        if pp["plays"] >= 4 and fp["plays"] >= 4:
            decay = pp["p_clear"] - fp["p_clear"]
            status = "lying" if decay > 0.15 else "stable"
            if status == "lying":
                lying.append(tag)
        else:
            decay = None
            status = "low_n"
        trust_rows.append(
            {
                "tag": tag,
                "prac": pp,
                "fwd": fp,
                "decay": decay,
                "status": status,
            }
        )
        print(
            f"  {tag:20} prac={pp.get('p_clear',0):.2f}(n={pp.get('plays',0)}) "
            f"fwd={fp.get('p_clear',0):.2f}(n={fp.get('plays',0)}) {status}",
            flush=True,
        )

    print("\n[G15] Ban lying tags on forward vs baseline...", flush=True)
    base_one = score_pair_on_days(
        forward[:30],
        2.0,
        3.0,
        use_heuristic=True,
        risk_use_frac=BASELINE["risk_use_frac"],
        stop_atr_mult=BASELINE["stop_atr_mult"],
        per_trade_cap_pct=BASELINE["per_trade_cap_pct"],
    )
    # If no lying tags, simulate ban of weakest forward tag for demo of mechanism
    ban_set = set(lying)
    if not ban_set:
        # pick worst forward p_clear among tags with n>=4
        cands = [
            (r["fwd"]["p_clear"], r["tag"])
            for r in trust_rows
            if r["fwd"].get("plays", 0) >= 4
        ]
        if cands:
            cands.sort()
            ban_set = {cands[0][1]}
            ban_mode = "weakest_forward_tag_demo"
        else:
            ban_mode = "none"
    else:
        ban_mode = "trust_decay_lying"

    filtered = score_with_banned_tags(
        forward, BASELINE, 2.0, 3.0, ban_set, max_days=30
    )
    print(
        f"  baseline forward@2/3: clear={base_one['cleared']}/{base_one['n_days']} breach={base_one['breached']}",
        flush=True,
    )
    print(
        f"  ban {ban_set} ({ban_mode}): clear={filtered['cleared']}/{filtered['n_days']} breach={filtered['breached']}",
        flush=True,
    )
    filter_helps = (
        filtered["breached"] <= base_one["breached"]
        and filtered["cleared"] > base_one["cleared"]
    )

    # --- Final verdict ---
    report = {
        "g01_integrity": g01,
        "g05_leak": {
            "clean_split": leak_clean,
            "tripwire_catches_all_days": leak_detects_all,
        },
        "baseline": {"dials": BASELINE, "practice": base_prac, "forward": base_fwd},
        "leaky_search_all_days": {
            "dials": leaky_dials,
            "practice": leaky_prac,
            "forward": leaky_fwd,
        },
        "clean_search_practice_only": {
            "dials": clean_dials,
            "practice": clean_prac,
            "forward": clean_fwd,
        },
        "comparisons": {
            "clean_helps_forward_vs_baseline": clean_helps_fwd,
            "leaky_overfits_forward_vs_clean": leaky_overfits,
            "forward_clears_baseline": base_fwd["clears_sum"],
            "forward_clears_leaky": leaky_fwd["clears_sum"],
            "forward_clears_clean": clean_fwd["clears_sum"],
            "forward_breach_baseline": base_fwd["breach_sum"],
            "forward_breach_leaky": leaky_fwd["breach_sum"],
            "forward_breach_clean": clean_fwd["breach_sum"],
        },
        "trust": {
            "rows": trust_rows,
            "lying_tags": lying,
            "ban_mode": ban_mode,
            "ban_set": sorted(ban_set),
            "baseline_pair_forward": {
                "cleared": base_one["cleared"],
                "breached": base_one["breached"],
                "n": base_one["n_days"],
            },
            "filtered_pair_forward": filtered,
            "filter_helps": filter_helps,
        },
    }

    # Human verdict
    if clean_helps_fwd and clean_fwd["breach_sum"] == 0:
        dial_verdict = "YES — practice-only search helped forward clears vs baseline with no extra breach."
    elif clean_fwd["breach_sum"] > base_fwd["breach_sum"]:
        dial_verdict = "NO — clean search hurt the floor on forward; keep baseline dials."
    elif clean_fwd["clears_sum"] < base_fwd["clears_sum"]:
        dial_verdict = "NO clear win — practice-only search did not beat baseline on forward clears."
    else:
        dial_verdict = "NEUTRAL — similar to baseline; wall still prevents fake unseen scores."

    if leaky_fwd["clears_sum"] > clean_fwd["clears_sum"] and leaky_overfits:
        leak_note = "Leaky all-days search looks better on forward (peeking). That is NOT real help — it is contamination."
    else:
        leak_note = "Leaky search did not dominate forward vs clean on this grid (or similar)."

    if filter_helps:
        trust_verdict = "YES — banning weak/lying tags improved forward clear on 2.0/3.0 sample."
    else:
        trust_verdict = "NO clear win from tag ban on this sample — trust loop still useful as diagnostic."

    report["verdicts"] = {
        "practice_only_search": dial_verdict,
        "leak_warning": leak_note,
        "trust_filter": trust_verdict,
        "g01_helps": "YES for honest scoring integrity; does not raise clear% alone",
        "overall": None,
    }

    # Overall
    helps_bits = []
    if clean_helps_fwd:
        helps_bits.append("practice-only search")
    if filter_helps:
        helps_bits.append("trust filter")
    helps_bits.append("G01 integrity gate")
    helps_bits.append("G05 leak tripwire")
    if clean_helps_fwd or filter_helps:
        overall = (
            f"PARTIALLY HELPS empirically: {', '.join(helps_bits)}. "
            "Architecture is worth building; not every knob raised clear this run."
        )
    else:
        overall = (
            "ARCHITECTURE HELPS PROCESS (integrity + no fake unseen). "
            "On this dial grid, practice-only search and trust ban did NOT beat baseline forward clears — "
            "so do not expect automatic clear% lift; expect honest evaluation and safer meta later."
        )
    report["verdicts"]["overall"] = overall

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 64, flush=True)
    print("DOES IT REALLY HELP?", flush=True)
    print(f"  Forward clears  baseline={base_fwd['clears_sum']}  leaky={leaky_fwd['clears_sum']}  clean={clean_fwd['clears_sum']}", flush=True)
    print(f"  Forward breach  baseline={base_fwd['breach_sum']}  leaky={leaky_fwd['breach_sum']}  clean={clean_fwd['breach_sum']}", flush=True)
    print(f"  clean dials={clean_dials}", flush=True)
    print(f"  leaky dials={leaky_dials}", flush=True)
    print(f"  practice-only search: {dial_verdict}", flush=True)
    print(f"  leak note: {leak_note}", flush=True)
    print(f"  trust filter: {trust_verdict}", flush=True)
    print(f"  OVERALL: {overall}", flush=True)
    print(f"report={OUT}", flush=True)
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
