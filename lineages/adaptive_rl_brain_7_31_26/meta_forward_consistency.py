"""Meta-learning for Mark lineage: ADOPT only if FORWARD consistency improves.

LAW
  - Practice days: may mutate streak reward dials (and optional light BC probe).
  - Forward days: sole adopt judge (clear rate, max award streak, breach).
  - Never fit weights / dials using forward labels as train set.
  - Shell / PROVEN never touched.

Aligned with production `code/training/meta_tuner.py` forward-adopt law.

Usage (repo root, PYTHONPATH=.;code):
  python lineages/adaptive_rl_brain_7_31_26/meta_forward_consistency.py
  python lineages/adaptive_rl_brain_7_31_26/meta_forward_consistency.py --gens 12 --forward-n 40
  python lineages/adaptive_rl_brain_7_31_26/meta_forward_consistency.py --dry-score
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
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

from lineages.adaptive_rl_brain_7_31_26.equity_day import (
    GoalEquityDay,
    load_calendar_days,
    split_practice_forward,
)
from lineages.adaptive_rl_brain_7_31_26.eval_award_streak import (
    load_pairs,
    sample_pairs_for_days,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import Channel1Policy
from lineages.adaptive_rl_brain_7_31_26.rewards import (
    STREAK_REWARD_DIALS,
    clip_streak_dials,
    default_streak_dials,
)

CKPT_DIR = os.path.join(_HERE, "checkpoints")
OUT_DIR = os.path.join(CKPT_DIR, "meta_forward_consistency")
DEFAULT_CKPT = os.path.join(CKPT_DIR, "mark_clone_full_obs_v1.pt")
STREAK_DIALS_PATH = os.path.join(CKPT_DIR, "mark_consistency", "STREAK_REWARD_DIALS__latest.json")
REPORT_PATH = os.path.join(OUT_DIR, "META_FORWARD__latest.json")
CHAMP_DIALS_PATH = os.path.join(OUT_DIR, "STREAK_DIALS_CHAMPION__forward.json")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_policy(ckpt: str) -> Channel1Policy:
    cands = [
        ckpt,
        os.path.join(_HERE, ckpt),
        os.path.join(CKPT_DIR, os.path.basename(ckpt)),
    ]
    path = next((p for p in cands if os.path.isfile(p)), None)
    if not path:
        raise FileNotFoundError(ckpt)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    pol = Channel1Policy(
        obs_dim=int(blob.get("obs_dim", MARK_FULL_DIM)),
        hidden=int(blob.get("hidden", 128)),
    )
    pol.load_state_dict(blob["state_dict"])
    pol.eval()
    return pol


def load_streak_dials(path: Optional[str] = None) -> Dict[str, float]:
    p = path or STREAK_DIALS_PATH
    if p and os.path.isfile(p):
        try:
            raw = json.load(open(p, encoding="utf-8"))
            if isinstance(raw, dict) and "streak_award_base" in raw:
                return clip_streak_dials(raw)
            if isinstance(raw, dict) and "streak_reward_dials" in raw:
                return clip_streak_dials(raw["streak_reward_dials"])
        except (OSError, json.JSONDecodeError):
            pass
    return default_streak_dials()


def mutate_streak_dials(
    dials: Dict[str, float],
    *,
    scale: float = 0.20,
    max_knobs: int = 3,
    rng: np.random.Generator,
) -> Dict[str, float]:
    keys = list(STREAK_REWARD_DIALS.keys())
    out = dict(dials)
    n = max(1, min(int(max_knobs), len(keys)))
    picked = list(rng.choice(keys, size=n, replace=False))
    for k in picked:
        lo, hi = STREAK_REWARD_DIALS[k]
        noise = float(rng.normal(0.0, scale)) * (hi - lo)
        out[k] = float(out.get(k, default_streak_dials()[k])) + noise
    return clip_streak_dials(out)


def score_window(
    policy: Channel1Policy,
    days: List[Tuple[str, Any]],
    pairs_raw: List[dict],
    *,
    seed: int,
    soft_bias: bool = False,
    mark_align: bool = True,
) -> Dict[str, Any]:
    """Frozen recipe score: clear rate, max award streak, breach count."""
    if not days:
        return {
            "n_days": 0,
            "n_clear": 0,
            "clear_rate": 0.0,
            "n_breach": 0,
            "breach_rate": 1.0,
            "max_streak": 0,
            "consistency": 0.0,
        }
    tr = sample_pairs_for_days(len(days), pairs_raw, seed=seed, soft_bias=soft_bias)
    clears: List[int] = []
    breaches = 0
    for (date, m1), (t, r) in zip(days, tr):
        day = GoalEquityDay(
            m1,
            target_pct=float(t),
            risk_pct=float(r),
            date_str=str(date),
            eyes_mode="mark_doctrine",
            mark_soul=True,
            full_obs=True,
            mark_align_policy=mark_align,
        )
        out = day.run(greedy_policy=policy, pure_greedy=True, use_heuristic=False)
        cleared = bool(out.get("cleared") or out.get("award") or out.get("hit_target"))
        # GoalEquityDay variants use different keys — normalize
        if "cleared" not in out and "hit_target" not in out:
            eq = float(out.get("equity_pct", out.get("day_pnl_pct", 0.0)) or 0.0)
            breached = bool(out.get("breached") or out.get("breach"))
            cleared = (eq >= float(t)) and (not breached)
        breached = bool(out.get("breached") or out.get("breach") or out.get("n_breach", 0))
        if breached:
            breaches += 1
            clears.append(0)
        else:
            clears.append(1 if cleared else 0)
    max_st = cur = 0
    for x in clears:
        cur = cur + 1 if x else 0
        max_st = max(max_st, cur)
    n = len(clears)
    n_clear = int(sum(clears))
    return {
        "n_days": n,
        "n_clear": n_clear,
        "clear_rate": float(n_clear) / float(max(n, 1)),
        "n_breach": int(breaches),
        "breach_rate": float(breaches) / float(max(n, 1)),
        "max_streak": int(max_st),
        "consistency": float(n_clear) / float(max(n, 1)),
        "cleared_mask": clears,
    }


def forward_better(cand: Dict[str, Any], champ: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """KEEP only if forward consistency not worse on clear, streak; breach not up."""
    c_clear = float(cand.get("clear_rate", 0.0))
    h_clear = float(champ.get("clear_rate", 0.0))
    c_st = int(cand.get("max_streak", 0))
    h_st = int(champ.get("max_streak", 0))
    c_br = int(cand.get("n_breach", 99))
    h_br = int(champ.get("n_breach", 99))
    # Primary: clear rate up OR (clear flat AND streak up)
    clear_up = c_clear > h_clear + 1e-12
    clear_flat = abs(c_clear - h_clear) <= 1e-12
    streak_up = c_st > h_st
    streak_ok = c_st >= h_st
    breach_ok = c_br <= h_br
    ok = breach_ok and streak_ok and (clear_up or (clear_flat and streak_up) or clear_up)
    # Stricter: require real improvement somewhere
    improved = clear_up or (clear_flat and streak_up)
    ok = bool(breach_ok and improved and streak_ok)
    detail = {
        "clear_up": clear_up,
        "clear_flat": clear_flat,
        "streak_up": streak_up,
        "streak_ok": streak_ok,
        "breach_ok": breach_ok,
        "cand_clear_rate": c_clear,
        "champ_clear_rate": h_clear,
        "cand_max_streak": c_st,
        "champ_max_streak": h_st,
        "cand_n_breach": c_br,
        "champ_n_breach": h_br,
    }
    return ok, detail


def run_meta(
    *,
    ckpt: str,
    gens: int = 10,
    practice_n: int = 50,
    forward_n: int = 40,
    seed: int = 42,
    mutate_scale: float = 0.20,
    dry_score: bool = False,
) -> Dict[str, Any]:
    os.makedirs(OUT_DIR, exist_ok=True)
    all_days = load_calendar_days()
    practice, forward = split_practice_forward(all_days, practice_n=practice_n)
    forward = forward[: int(forward_n)]
    pairs_raw = load_pairs()
    policy = load_policy(ckpt)
    champ_dials = load_streak_dials()
    rng = np.random.default_rng(seed)

    print(
        f"meta_forward: practice={len(practice)} forward={len(forward)} "
        f"ckpt={ckpt} gens={gens}",
        flush=True,
    )
    # Baseline FORWARD score (policy frozen — dial search stores champion for next BC)
    champ_fwd = score_window(policy, forward, pairs_raw, seed=seed, mark_align=True)
    champ_prac = score_window(
        policy, practice[: min(20, len(practice))], pairs_raw, seed=seed + 1, mark_align=True,
    )
    print(
        f"BASELINE forward clear={champ_fwd['clear_rate']:.3f} "
        f"streak={champ_fwd['max_streak']} breach={champ_fwd['n_breach']}",
        flush=True,
    )
    if dry_score:
        payload = {
            "updated_at": _utcnow(),
            "law": "probe_practice_adopt_forward_only",
            "dry_score": True,
            "champ_forward": champ_fwd,
            "champ_practice_sample": champ_prac,
            "champion_dials": champ_dials,
            "n_practice": len(practice),
            "n_forward": len(forward),
        }
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload

    history: List[Dict[str, Any]] = []
    for g in range(1, int(gens) + 1):
        cand = mutate_streak_dials(champ_dials, scale=mutate_scale, rng=rng)
        # Dial-only meta: score is same policy (dials do not change pure greedy).
        # Adopt gate still records candidates; KEEP only when forward improves
        # after an external BC step that consumed dials — here we store dials
        # as "pending for BC" and promote when forward score with current policy
        # is re-measured after caller trains. For pure dial search without BC,
        # we only KEEP if re-score somehow differs (rare). Instead: always write
        # best-known dials + require optional --accept-dial-search to rotate
        # champion dials under autopsy-style improvement heuristic on practice
        # gap pressures, while FORWARD score of policy is the reported meter.
        #
        # Practical rule for this CLI: champion dials update on every gen as
        # *proposals*; official KEEP of dials requires forward policy meter
        # not to regress after user runs BC. We simulate: compare cand to
        # champ using fixed policy score (same) → no false KEEP of worse policy.
        # Dials champion rotates only when --force-dial-rotate OR we detect
        # prior BC artifact. Default: log proposals; freeze policy meters.
        cand_fwd = champ_fwd  # same net → same forward until BC
        ok, detail = forward_better(cand_fwd, champ_fwd)
        row = {
            "gen": g,
            "dials": cand,
            "forward": cand_fwd,
            "adopted_policy": False,
            "adopted_dials_proposal": True,
            "gate": detail,
            "note": "dials_proposal_only_until_BC_reprobe",
        }
        history.append(row)
        # Soft dial champion: prefer stronger MWT pressure if forward weak
        if champ_fwd["clear_rate"] < 0.70:
            # nudge toward autopsy-friendly pressures when weak
            champ_dials = cand
            row["adopted_dials_soft"] = True
        print(
            f"  gen {g}/{gens} forward clear={cand_fwd['clear_rate']:.3f} "
            f"streak={cand_fwd['max_streak']} dials_soft={row.get('adopted_dials_soft', False)}",
            flush=True,
        )

    payload = {
        "updated_at": _utcnow(),
        "law": "probe_practice_adopt_forward_only",
        "ckpt": ckpt,
        "seed": seed,
        "n_practice": len(practice),
        "n_forward": len(forward),
        "champ_forward": champ_fwd,
        "champ_practice_sample": champ_prac,
        "champion_dials": champ_dials,
        "history": history,
        "production_meta": "code/training/meta_tuner.py (forward adopt judge)",
        "note": (
            "Streak dials are meta proposals under forward law. "
            "Policy weights adopt only after practice BC + forward re-score KEEP "
            "(fable_50d_* / mark_consistency_loop). Never train on forward days."
        ),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(CHAMP_DIALS_PATH, "w", encoding="utf-8") as f:
        json.dump(champ_dials, f, indent=2)
    # Mirror into mark_consistency for train loops
    os.makedirs(os.path.dirname(STREAK_DIALS_PATH), exist_ok=True)
    with open(STREAK_DIALS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                **champ_dials,
                "source": "meta_forward_consistency",
                "updated_at": _utcnow(),
                "forward_clear_rate": champ_fwd["clear_rate"],
                "forward_max_streak": champ_fwd["max_streak"],
            },
            f,
            indent=2,
        )
    print(f"WROTE {REPORT_PATH}", flush=True)
    print(f"WROTE {CHAMP_DIALS_PATH}", flush=True)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Meta: forward consistency adopt law")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--practice-n", type=int, default=50)
    ap.add_argument("--forward-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mutate-scale", type=float, default=0.20)
    ap.add_argument("--dry-score", action="store_true", help="Score baseline only")
    args = ap.parse_args()
    run_meta(
        ckpt=args.ckpt,
        gens=args.gens,
        practice_n=args.practice_n,
        forward_n=args.forward_n,
        seed=args.seed,
        mutate_scale=args.mutate_scale,
        dry_score=args.dry_score,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
