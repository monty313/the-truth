"""Grow up like Mark — you lead; he must learn to learn (forward adult).

Mark's process (what we teach as *how*, not day answers):
  see multi-TF → wait loaded → fire with size → name the error class →
  reweight attention → KEEP/REJECT conscience → prove on NEW days (no retrain).

Child = copy bars. Teen = path classes + spine heads. Adult = principles + forward gate.

Runs principle adult cycle (practice train, forward verify, learn≠copy) and
writes GROW_UP report. Optionally notes L2L memory + spine shadow products.

Doctrine: Spine Shadow + pt5 learn≠copy + UNSEEN forward adopt.
PROVEN never touched.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_ROOT), str(_ROOT / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = _HERE / "checkpoints" / "fable_50d_match"
GROW_MD = OUT / "GROW_UP_MARK_STYLE__latest.md"
GROW_JSON = OUT / "GROW_UP_MARK_STYLE__latest.json"
BEST = OUT / "BEST__latest.json"
L2L_MEM = OUT / "L2L_PATH_MEMORY.jsonl"

MARK_HOW = """
## How Mark learns (the process the student must internalize)

| # | Mark | Student learns | Code product |
|---|------|----------------|--------------|
| 1 | Reads HTF then LTF | Side only with force permission | force-gate / mark_align |
| 2 | Waits loaded, not frozen | phase before_first_fire + wait_loaded | shadow phase/event heads |
| 3 | Fires with size to goal | fire/add + size_bucket | shadow size head |
| 4 | Names the mistake class | early/late/thrash/miss — not the date | L2L path classes + memory |
| 5 | Ignores lying sensors | attention / who to trust | clue_gate |
| 6 | Undoes bad days | if pack dies, restore | KEEP/REJECT |
| 7 | Same mind on new days | no retrain at score | forward principle + score_forward_100d |

**You lead** with principles, HITL spines, and conscience.  
**He grows up** when forward works without fitting those days.
"""


def _load_json(p: Path) -> dict:
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def run_adult_principle_cycle(*, epochs: int = 40, seed: int = 42) -> Dict[str, Any]:
    from lineages.adaptive_rl_brain_7_31_26.forward_principle_learn.train_principle_student import (
        run_forward_learn_cycle,
    )

    return run_forward_learn_cycle(seed=seed, epochs=epochs, write=True)


def l2l_memory_summary() -> Dict[str, Any]:
    if not L2L_MEM.is_file():
        return {"n": 0, "note": "run learn_to_learn_path.py for path-class memory"}
    rows = []
    for line in L2L_MEM.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    dom: Dict[str, int] = {}
    for r in rows[-16:]:
        c = r.get("dominant_class")
        if c:
            dom[c] = dom.get(c, 0) + 1
    return {"n": len(rows), "recent_dominant_classes": dom, "last": rows[-1] if rows else None}


def write_report(principle: Dict[str, Any], l2l: Dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    best = _load_json(BEST)
    # extract gates from principle cycle report shape
    gates = principle.get("gates") or principle.get("gate_reports") or {}
    if isinstance(gates, list):
        gate_map = {g.get("gate"): g for g in gates if isinstance(g, dict)}
    else:
        gate_map = gates if isinstance(gates, dict) else {}
    lnc = gate_map.get("learn_not_copy") or principle.get("learn_not_copy") or {}
    if hasattr(lnc, "to_dict"):
        lnc = lnc.to_dict()
    adult = bool(principle.get("adopt") or principle.get("keep") or principle.get("adult_pass"))
    # common keys from run_forward_learn_cycle
    prac = (
        principle.get("practice_score")
        or principle.get("practice")
        or principle.get("practice_metrics")
        or {}
    )
    fwd = (
        principle.get("forward_score")
        or principle.get("forward")
        or principle.get("forward_metrics")
        or {}
    )
    cold = principle.get("cold_forward") or {}
    decision = principle.get("decision") or ("KEEP" if adult else "REJECT")
    if principle.get("decision") == "KEEP" or principle.get("promote"):
        adult = True
    # gates nested
    gobj = principle.get("gates") or {}
    if isinstance(gobj, dict) and "gates" in gobj:
        for g in gobj.get("gates") or []:
            if g.get("gate") == "learn_not_copy":
                lnc = g
                break

    body = f"""# Grow up like Mark — learn to learn for forward

**When:** {datetime.now(timezone.utc).isoformat()}  
**Child body (50d pack):** same={best.get('same_outcome')} mwt={best.get('mwt')} breach={best.get('breach')}  
**Adult principle cycle decision:** **{decision}**  
**learn≠copy:** {lnc}

{MARK_HOW}

## Adult cycle result (principles — not day IDs)

| Split | Metrics |
|-------|---------|
| Practice | `{json.dumps(prac, default=str)[:500]}` |
| **Forward (new tasks, no fit)** | `{json.dumps(fwd, default=str)[:500]}` |
| Cold forward (untrained baseline) | `{json.dumps(cold, default=str)[:300]}` |
| Decision | {decision} |

**Meaning:**  
- If decision is KEEP/adopt → he internalized principles enough to **transfer** (adult step).  
- If REJECT → still a child/teen on principles; do **not** promote day-oracle spam.

## Path-class memory (mistake vocabulary)

- Rows: {l2l.get('n')}  
- Recent dominant classes: {l2l.get('recent_dominant_classes')}  
- These are *how Mark names errors* so the same early-fire bug is one skill, not 15 day memos.

## Spine Shadow heads (day structure)

Already shipped in `mark_shadow_policy.py`:

| Head | Adult skill |
|------|-------------|
| phase | where am I in the day |
| event | wait / fire / add / hold |
| size | how hard |
| clue_gate | who to trust (meta) |

Train: `train_spine_shadow_full.py` · Path L2L: `learn_to_learn_path.py`

## Your role (Mark) vs his role (student)

| You lead | He must do alone later |
|----------|-------------------------|
| Principles, HITL spines, KEEP conscience | Decide under force-gate |
| Name error classes on chart | Apply class on **unseen** days |
| Refuse pack-killing updates | Self-limit thrash (HOLD skill) |
| Forward exam | No retrain at score time |

## Stages

| Stage | What he does | Forward |
|-------|--------------|---------|
| Child | Act-only copy practice days | Weak |
| Teen | Path classes + multi-head spine | Improving if pack safe |
| **Adult** | Principles + attention + KEEP internalized + **forward gate PASS** | **Goal** |

## Commands

```powershell
cd C:\\Users\\user\\Fable5_Foundation\\MOMENTUM_ONE\\the-truth
$env:PYTHONPATH = ".;code"

# 1) Adult principles (this script)
python lineages/adaptive_rl_brain_7_31_26/grow_up_mark_style.py

# 2) Path mistake vocabulary (meta boost)
python lineages/adaptive_rl_brain_7_31_26/learn_to_learn_path.py --max-rounds 6

# 3) Full day structure heads
python lineages/adaptive_rl_brain_7_31_26/train_spine_shadow_full.py --max-rounds 4

# 4) Honest adult exam — calendar holdout
python lineages/adaptive_rl_brain_7_31_26/score_forward_100d.py --n-days 100 --partial 20
```

## Why this is “like I learn”

You do not re-memorize every tick of every past day when the market is new.  
You re-use **principles**, **mistake types**, and **attention**.  
That is learn-to-learn. That is what survives **forward testing**.

Day-oracle BC alone raises practice then dies forward and kills the pack — we already measured that (35→32).  
Adult path: principles + classes + clue_gate + conscience + forward gate.
"""
    GROW_MD.write_text(body, encoding="utf-8")
    GROW_JSON.write_text(
        json.dumps(
            {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "best_50d": best,
                "principle_cycle": {
                    k: principle.get(k)
                    for k in (
                        "decision",
                        "practice",
                        "forward",
                        "gates",
                        "adopt",
                        "keep",
                        "saved_at",
                    )
                    if k in principle or True
                },
                "principle_raw_keys": list(principle.keys())[:40],
                "l2l": l2l,
                "adult_like": adult or decision == "KEEP",
                "proven_touched": False,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    for dest in (
        Path(r"C:\Users\user\OneDrive\Desktop\ARMY\01_SYSTEM\outputs\army\GROW_UP_MARK_STYLE__latest.md"),
        Path(r"C:\Users\user\AppData\Local\Temp\grok-goal-a7f5320040c5\implementer\GROW_UP_MARK_STYLE__latest.md"),
    ):
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body, encoding="utf-8")
        except OSError:
            pass
    print(body[:3500], flush=True)
    print(f"\nWROTE {GROW_MD}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print("=== GROW UP like Mark (learn to learn) ===", flush=True)
    try:
        principle = run_adult_principle_cycle(epochs=args.epochs, seed=args.seed)
        print(
            f"principle decision={principle.get('decision')} "
            f"keys={list(principle.keys())[:12]}",
            flush=True,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        principle = {"decision": "REJECT", "error": str(e)}
    l2l = l2l_memory_summary()
    write_report(principle, l2l)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
