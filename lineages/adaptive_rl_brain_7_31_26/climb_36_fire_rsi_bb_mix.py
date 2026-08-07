"""Teen 36+ climb: proven fire_skill multi-day Mark pool MIXED with RSI+BB timing.

Recipe
------
PRIMARY (what got 35→36):
  multi-day Mark path-family **fire_skill** across all MWT days
  (miss_continuation + ltf_continuation) — not calendar memos

TIMING SKILL (user doctrine / next L2L):
  RSI(5)+BB on RSI on LTF under HTF price-BB mass
  → continuation fires + light pullback waits (all official sets)

PROTECT:
  award self-imitate · HOLD floor · high KL to live BEST · Fable KEEP only if same↑

Usage:
  cd the-truth; $env:PYTHONPATH=".;code"
  python -u lineages/adaptive_rl_brain_7_31_26/climb_36_fire_rsi_bb_mix.py --max-rounds 8
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

from lineages.adaptive_rl_brain_7_31_26.equity_day import load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.fable_50d_mark_match_loop import load_policy, save_policy
from lineages.adaptive_rl_brain_7_31_26.fable_50d_rapid import award_self, load_oracle, score_policy
from lineages.adaptive_rl_brain_7_31_26.fable_kag_l2l import collect_cluster_labels
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_HOLD, Channel1Policy
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials
from lineages.adaptive_rl_brain_7_31_26.rsi_bb_l2l_skill import build_rsi_bb_skill_batch
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import train_bc

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
TEEN = os.path.join(_HERE, "checkpoints", "TEEN_STAGE_same36_fable_kag_fire_skill.pt")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")
BEST = os.path.join(OUT, "BEST__latest.json")
HARNESS = os.path.join(OUT, "CLIMB36_FIRE_RSIBB_MIX__latest.json")
REPORT = os.path.join(OUT, "CLIMB36_FIRE_RSIBB_MIX__latest.md")
MEM = os.path.join(OUT, "CLIMB36_FIRE_RSIBB_MIX_MEMORY.jsonl")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stack(
    parts: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ws = [], [], []
    for X, y, w in parts:
        if len(y) == 0:
            continue
        xs.append(X)
        ys.append(y)
        ws.append(w)
    if not xs:
        z = np.zeros((0, MARK_FULL_DIM), np.float32)
        e = np.zeros((0,), np.int64)
        return z, e, np.zeros((0,), np.float32)
    X = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int64)
    w = np.concatenate(ws, axis=0).astype(np.float32)
    w = np.maximum(w, 1e-6)
    w = w / float(w.mean())
    return X, y, w


def inject_hold_awards(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    day_map: Dict[str, Any],
    awards: List[dict],
    policy: Channel1Policy,
    *,
    min_hold: float = 0.38,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    hold_frac = float((y == 0).mean()) if len(y) else 0.0
    if hold_frac >= min_hold:
        return X, y, w, hold_frac
    xs, ys, ws = list(X), list(y), list(w)
    for row in awards[:32]:
        a, b, c = award_self(
            day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"]), policy
        )
        for o, act, ww in zip(a, b, c):
            if int(act) == ACTION_HOLD:
                xs.append(o)
                ys.append(0)
                ws.append(float(ww) * 3.0)
    X2 = np.stack(xs).astype(np.float32)
    y2 = np.asarray(ys, np.int64)
    w2 = np.asarray(ws, np.float32)
    w2 = w2 / float(w2.mean())
    return X2, y2, w2, float((y2 == 0).mean())


def run_mix(
    *,
    max_rounds: int = 8,
    keep_floor: int = 36,
    epochs: int = 16,
    kl_coef: float = 1.20,
    lr: float = 1.1e-4,
    fire_weight: float = 1.0,
    rsi_cont_weight: float = 0.45,
    rsi_pull_weight: float = 0.18,
    seed: int = 42,
) -> Dict[str, Any]:
    os.makedirs(OUT, exist_ok=True)
    print("=== CLIMB 36+ MIX: fire_skill Mark + RSI-BB LTF timing ===", flush=True)
    print(
        f"  mix weights fire={fire_weight} rsi_cont={rsi_cont_weight} "
        f"rsi_pull={rsi_pull_weight} kl={kl_coef}",
        flush=True,
    )

    # Always start from teen KEEP if present
    if os.path.isfile(TEEN):
        src = TEEN
        # keep live aligned unless another job owns it
        try:
            import shutil

            shutil.copy2(TEEN, CKPT)
        except OSError:
            pass
    else:
        src = CKPT
    print(f"  load {os.path.basename(src)}", flush=True)
    policy = load_policy(src)
    assert not getattr(policy, "multi_head", False)

    baseline = json.load(open(BASELINE, encoding="utf-8"))
    mark_rows = baseline["rows"]
    floor_clear = int(baseline.get("policy_clear", 27))
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
    day_map = {str(d): m1 for d, m1 in days}
    all_days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)
    oracle = load_oracle()
    dials = clip_streak_dials(default_streak_dials())

    print("Score base…", flush=True)
    best = score_policy(policy, day_map, mark_rows)
    print(
        f"START same={best['same_outcome']} mwt={best['mark_would_take']} "
        f"breach={best['n_breach']} clear={best['policy_clear']}",
        flush=True,
    )
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    live_floor = max(keep_floor, int(best["same_outcome"]))
    cycles: List[dict] = []

    hid = 128
    try:
        w0 = best_state.get("net.0.weight")
        if w0 is not None:
            hid = int(w0.shape[0])
    except Exception:
        hid = 128

    # Precompute RSI-BB skill pools once (timing teacher) — reuse each round
    print("Build RSI-BB skill pools (LTF timing under HTF mass)…", flush=True)
    Xc, yc, _tc, _wc, wc, meta_c = build_rsi_bb_skill_batch(
        all_days,
        kind="continuation",
        max_days=45,
        seed=seed,
        mark_rows=mark_rows,
        day_map=day_map,
        oracle=oracle,
        concurrence_boost=2.0,
        base_w_cont=1.0,
    )
    Xp, yp, _tp, _wp, wp, meta_p = build_rsi_bb_skill_batch(
        all_days,
        kind="pullback",
        max_days=45,
        seed=seed + 1,
        mark_rows=mark_rows,
        day_map=day_map,
        oracle=oracle,
        concurrence_boost=1.6,
        base_w_pull=1.0,
    )
    print(
        f"  rsi_cont n={meta_c.get('n_keep')} sets={meta_c.get('sets_with_hits')} "
        f"conc={meta_c.get('concurrence_n')}",
        flush=True,
    )
    print(
        f"  rsi_pull n={meta_p.get('n_keep')} sets={meta_p.get('sets_with_hits')} "
        f"conc={meta_p.get('concurrence_n')}",
        flush=True,
    )

    # Round recipes: mostly fire-primary mix; one round cont-heavy RSI
    recipes = [
        "fire+rsi_cont",
        "fire+rsi_both",
        "fire+rsi_cont",
        "rsi_cont_heavy+fire",
        "fire+rsi_both",
        "fire+rsi_cont",
        "fire+rsi_both",
        "fire+rsi_cont",
    ]

    for rnd in range(1, max_rounds + 1):
        recipe = recipes[(rnd - 1) % len(recipes)]
        print(f"\n===== MIX {rnd}/{max_rounds} recipe={recipe} =====", flush=True)
        policy.load_state_dict(best_state)
        mwt = [r for r in best["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"]
        awards = [r for r in best["rows"] if r["miss_class"] == "AWARD"]
        if not mwt:
            print("no MWT — done", flush=True)
            break

        # PRIMARY: multi-day Mark fire_skill (proven 35→36)
        Xf, yf, wf, meta_f = collect_cluster_labels(
            day_map, mwt, awards, policy, oracle, target_family="fire_skill"
        )
        print(f"  fire_skill meta={meta_f}", flush=True)
        if meta_f["n"] < 30:
            Xf, yf, wf, meta_f = collect_cluster_labels(
                day_map, mwt, awards, policy, oracle, target_family="all"
            )
            print(f"  fallback all meta={meta_f}", flush=True)

        parts: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        sources: List[str] = []

        if meta_f["n"] >= 30:
            parts.append((Xf, yf, wf * float(fire_weight)))
            sources.append("mark_fire_skill_multiday")

        if "rsi_cont" in recipe or "rsi_both" in recipe or "rsi_cont_heavy" in recipe:
            if len(yc) > 0:
                cw = float(rsi_cont_weight)
                if "rsi_cont_heavy" in recipe:
                    cw = float(rsi_cont_weight) * 1.6
                parts.append((Xc, yc, wc * cw))
                sources.append("rsi_bb_continuation_ltf")

        if "rsi_both" in recipe and len(yp) > 0:
            # light pullback only — wait-alone pack-killed before
            # subsample to avoid HOLD flood
            n_pull = min(len(yp), max(40, len(yc) // 2 if len(yc) else 80))
            rng = np.random.default_rng(seed + rnd)
            idx = rng.choice(len(yp), size=n_pull, replace=False)
            parts.append(
                (
                    Xp[idx],
                    yp[idx],
                    wp[idx] * float(rsi_pull_weight),
                )
            )
            sources.append("rsi_bb_pullback_ltf_light")

        X, y, w = _stack(parts)
        if len(y) < 50:
            print("  sparse mix — skip", flush=True)
            continue

        X, y, w, hold_frac = inject_hold_awards(
            X, y, w, day_map, awards, policy, min_hold=0.38
        )
        print(
            f"  mix n={len(y)} hold_frac={hold_frac:.3f} dir={int((y!=0).sum())} "
            f"sources={sources}",
            flush=True,
        )

        pol2, losses = train_bc(
            X,
            y,
            sample_weights=w,
            epochs=epochs,
            lr=lr,
            hidden=hid,
            seed=1000 + rnd,
            warm_state=best_state,
            kl_anchor_state=best_state,
            kl_coef=float(kl_coef),
            freeze_trunk=False,
        )
        with torch.no_grad():
            pred = pol2(torch.tensor(X, dtype=torch.float32)).argmax(-1).numpy()
        act_match = float((pred == y).mean())
        print(
            f"  train act_match={act_match:.3f} loss={losses[-1] if losses else 0:.4f}",
            flush=True,
        )

        post = score_policy(pol2, day_map, mark_rows)
        print(
            f"  POST same={post['same_outcome']} mwt={post['mark_would_take']} "
            f"breach={post['n_breach']} clear={post['policy_clear']}",
            flush=True,
        )

        keep = (
            post["n_breach"] == 0
            and post["policy_clear"] >= floor_clear
            and post["same_outcome"] >= live_floor
            and post["same_outcome"] > best["same_outcome"]
        )
        decision = "KEEP" if keep else "REJECT"
        if post["same_outcome"] < best["same_outcome"] - 2:
            decision = "REJECT"
            print("  pack crater — REJECT", flush=True)
        elif not keep:
            print("  REJECT", flush=True)

        if decision == "KEEP":
            best = post
            best_state = {k: v.detach().clone() for k, v in pol2.state_dict().items()}
            live_floor = max(live_floor, int(post["same_outcome"]))
            save_policy(
                pol2,
                note=f"climb36_mix_KEEP_r{rnd}_{recipe}",
                dials=dials,
            )
            # refresh teen backup if new high water
            try:
                import shutil

                shutil.copy2(CKPT, TEEN.replace("same36", f"same{post['same_outcome']}"))
            except OSError:
                pass
            with open(BEST, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "same_outcome": post["same_outcome"],
                        "policy_clear": post["policy_clear"],
                        "mwt": post["mark_would_take"],
                        "breach": post["n_breach"],
                        "source": f"climb36_fire_rsibb_mix_KEEP_r{rnd}",
                        "stage": "teen",
                        "growth_method": "climb36_fire_rsi_bb_mix",
                        "core_skill": (
                            "mark fire_skill multi-day + LTF RSI-BB timing under HTF mass"
                        ),
                        "recipe": recipe,
                        "sources": sources,
                        "note": "Mix of proven 36 recipe + RSI-BB LTF timing L2L",
                        "ts": _utcnow(),
                    },
                    f,
                    indent=2,
                )
            print(f"  KEEP best_same={best['same_outcome']}", flush=True)
            try:
                with open(os.path.join(OUT, "WHAT_WORKS__GOAL.md"), "a", encoding="utf-8") as wf:
                    wf.write(
                        f"| KEEP fire+rsi mix | **{best['same_outcome']}** | "
                        f"{best['mark_would_take']} | {best['n_breach']} | "
                        f"{recipe} |\n"
                    )
            except OSError:
                pass
        else:
            policy.load_state_dict(best_state)

        row = {
            "ts": _utcnow(),
            "round": rnd,
            "recipe": recipe,
            "sources": sources,
            "decision": decision,
            "same": post["same_outcome"],
            "mwt": post["mark_would_take"],
            "breach": post["n_breach"],
            "best_same": best["same_outcome"],
            "act_match": act_match,
            "fire_meta": meta_f,
            "method": "climb36_fire_rsi_bb_mix",
        }
        cycles.append(row)
        with open(MEM, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        with open(HARNESS, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "updated_at": _utcnow(),
                    "best_same": best["same_outcome"],
                    "best_mwt": best["mark_would_take"],
                    "best_breach": best["n_breach"],
                    "method": "climb36_fire_rsi_bb_mix",
                    "mix": {
                        "fire_weight": fire_weight,
                        "rsi_cont_weight": rsi_cont_weight,
                        "rsi_pull_weight": rsi_pull_weight,
                    },
                    "rsi_meta": {"cont": meta_c, "pull": meta_p},
                    "cycles": cycles[-20:],
                },
                f,
                indent=2,
            )
        with open(REPORT, "w", encoding="utf-8") as f:
            f.write(
                f"# Climb 36+ fire_skill + RSI-BB mix\n\n"
                f"- best_same: **{best['same_outcome']}**\n"
                f"- last: {decision} recipe={recipe} same={post['same_outcome']}\n"
                f"- primary: Mark multi-day fire_skill (what got 36)\n"
                f"- timing: RSI+BB on LTF under HTF mass\n"
            )

        if best["same_outcome"] >= 50 and best["n_breach"] == 0:
            break

    summary = {
        "best_same": best["same_outcome"],
        "best_mwt": best["mark_would_take"],
        "best_breach": best["n_breach"],
        "cycles": len(cycles),
        "method": "climb36_fire_rsi_bb_mix",
        "raised": best["same_outcome"] > keep_floor,
    }
    print(f"DONE mix {summary}", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--keep-floor", type=int, default=36)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--kl-coef", type=float, default=1.20)
    ap.add_argument("--lr", type=float, default=1.1e-4)
    ap.add_argument("--fire-weight", type=float, default=1.0)
    ap.add_argument("--rsi-cont-weight", type=float, default=0.45)
    ap.add_argument("--rsi-pull-weight", type=float, default=0.18)
    args = ap.parse_args()
    run_mix(
        max_rounds=args.max_rounds,
        keep_floor=args.keep_floor,
        epochs=args.epochs,
        kl_coef=args.kl_coef,
        lr=args.lr,
        fire_weight=args.fire_weight,
        rsi_cont_weight=args.rsi_cont_weight,
        rsi_pull_weight=args.rsi_pull_weight,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
