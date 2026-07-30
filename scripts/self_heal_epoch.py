#!/usr/bin/env python3
"""One self-correct / self-improve epoch for Momentum One.

Closed loop (SkillOpt-style + CMO doctrine):
  1) Trajectories  — Mind Probe + Ghosts + aggregate IRAC
  2) Skill proposal — evidence bullet + optional reward nudge
  3) Optional short frontier train (consistency_sprint)
  4) prove_it gate — clear% must not fall; breach must stay 0
  5) Accept skill (+ optional rewards.yaml nudge) OR reject

Does not retrain core weights from scratch. Does not expand obs.

Usage:
  python scripts/self_heal_epoch.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
  python scripts/self_heal_epoch.py <brain> 3.0 3.5 --sprint-minutes 30 --auto-accept-skill --apply-reward-nudge
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKILL = ROOT / "references" / "doctrine" / "policy_skill.md"
BEST = ROOT / "artifacts" / "skills" / "best_skill.md"
REJECTED = ROOT / "artifacts" / "skills" / "rejected"
EPOCH_DIR = ROOT / "artifacts" / "self_heal_epochs"
REWARDS = ROOT / "configs" / "rewards.yaml"

PULLBACK_BOUNDS = (0.05, 0.50)
# Self-heal toolkit dials — search ranges (meta/self_heal may move; defaults in yaml are 0)
DIAL_BOUNDS = {
    "w_pullback_with_htf": (0.05, 0.50),
    "w_with_trend_close": (0.0, 1.0),
    "w_against_trend_close": (-1.0, 0.0),
    "w_setup_skip": (-0.5, 0.0),
    "w_quick_pull_close": (0.0, 0.5),
    "w_did_nothing": (-25.0, 0.0),
}


def run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(
        cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=timeout
    )


def parse_prove_it(stdout: str) -> dict:
    out = {"clear_pct": None, "breach_pct": None, "row": None, "raw": stdout}
    m = re.search(r"cleared.*?(\d+(?:\.\d+)?)\s*%", stdout, re.I)
    if m:
        out["clear_pct"] = float(m.group(1))
    m = re.search(r"breached.*?(\d+(?:\.\d+)?)\s*%", stdout, re.I)
    if m:
        out["breach_pct"] = float(m.group(1))
    m = re.search(r"longest cleared streak.*?(\d+)\s*days", stdout, re.I)
    if m:
        out["row"] = int(m.group(1))
    return out


def load_irac(brain: str) -> dict | None:
    path = ROOT / "artifacts" / "llm_curriculum" / f"irac_{brain}.json"
    if not path.is_file():
        matches = list((ROOT / "artifacts" / "llm_curriculum").glob("irac_*.json"))
        if not matches:
            return None
        path = max(matches, key=lambda p: p.stat().st_mtime)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def propose_from_irac(irac: dict) -> dict:
    """Map IRAC metrics → search directions (dials), not frozen final answers."""
    app = irac.get("application", {})
    conc = irac.get("conclusion", {})
    ph = int(app.get("sum_policy_hold_on_setup") or 0)
    hm = int(app.get("sum_high_miss_pull") or 0)
    wrong_bull = int(app.get("sum_wrong_side_under_bull") or 0)
    wrong_bear = int(app.get("sum_wrong_side_under_bear") or 0)
    n_cb = int(app.get("sum_cont_buy_only") or 0)
    n_cs = int(app.get("sum_cont_sell_only") or 0)
    side_bull = float(app.get("mean_side_bias_bull") or 0.0)
    side_bear = float(app.get("mean_side_bias_bear") or 0.0)
    mask_v = int(app.get("sum_mask_veto") or 0)
    cls = conc.get("class") or "Policy"

    # Prefer measured class from side metrics when present
    if wrong_bull + wrong_bear > max(20, (ph // 2)) and (side_bull < 0.02 or side_bear < 0.02):
        cls = "WrongSide"
    elif mask_v > ph and mask_v > 20:
        cls = "Shell"
    elif ph > 50 or hm > 10:
        cls = cls if cls in ("Policy", "Perception") else "Policy"

    proposal = {
        "class": cls,
        "skill_bullet": (
            f"{date.today()}: class={cls}; policy_hold={ph}; high_miss_pull={hm}; "
            f"wrong_side bull/bear={wrong_bull}/{wrong_bear}; "
            f"side_bias bull/bear={side_bull:+.3f}/{side_bear:+.3f}; "
            f"search dials per class (do not freeze human answer)."
        ),
        "reward_nudge": None,
        "reward_nudges": [],
        "rationale": irac.get("issue", ""),
    }

    if cls == "Shell":
        proposal["skill_bullet"] += " Mask veto dominates — do not chase rewards; inspect Shell."
        return proposal

    if cls == "WrongSide" or (wrong_bull > 30 and side_bull < 0.05):
        proposal["reward_nudges"] = [
            {"key": "w_with_trend_close", "delta": 0.05,
             "why": "wrong_side under bull cont — try with-trend close dial"},
            {"key": "w_against_trend_close", "delta": -0.05,
             "why": "penalize against-trend closes (search toward more negative)"},
        ]
        proposal["reward_nudge"] = proposal["reward_nudges"][0]
        proposal["skill_bullet"] = (
            f"{date.today()}: WrongSide — when wrong_side_under_bull high / side_bias_bull low, "
            f"search ↑ w_with_trend_close and ↓ w_against_trend_close; prove_it gate."
        )
    elif cls == "Policy" and (ph > 50 or hm > 10):
        proposal["reward_nudges"] = [
            {"key": "w_pullback_with_htf", "delta": 0.05,
             "why": "policy_hold on visible pull/cont — strengthen bread-and-butter"},
            {"key": "w_setup_skip", "delta": -0.01,
             "why": "optional: tax hold when setup visible (dial default 0)"},
        ]
        proposal["reward_nudge"] = proposal["reward_nudges"][0]
    elif cls == "Perception" and hm > 10:
        proposal["reward_nudges"] = [
            {"key": "w_pullback_with_htf", "delta": 0.05,
             "why": "low alt_prob on pull — amplify pull recognition payoff"},
        ]
        proposal["reward_nudge"] = proposal["reward_nudges"][0]

    _ = (n_cb, n_cs)  # available for future density-aware proposals
    return proposal


def read_reward(key: str):
    if not REWARDS.is_file():
        return None
    text = REWARDS.read_text(encoding="utf-8")
    m = re.search(rf"^{re.escape(key)}:\s*([-+0-9.eE]+)", text, re.M)
    return float(m.group(1)) if m else None


def apply_reward_nudge(key: str, delta: float):
    cur = read_reward(key)
    if cur is None:
        return False, f"{key} not found"
    new = cur + delta
    if key in DIAL_BOUNDS:
        lo, hi = DIAL_BOUNDS[key]
        new = max(lo, min(hi, new))
    elif key == "w_pullback_with_htf":
        lo, hi = PULLBACK_BOUNDS
        new = max(lo, min(hi, new))
    text = REWARDS.read_text(encoding="utf-8")
    text2, n = re.subn(
        rf"^{re.escape(key)}:\s*[-+0-9.eE]+",
        f"{key}: {new:.4f}",
        text,
        count=1,
        flags=re.M,
    )
    if n != 1:
        return False, "replace failed"
    stamp = f"# - {date.today()}  {key} {cur:.4f}->{new:.4f} — WHY: self_heal_epoch gated nudge\n"
    if stamp not in text2:
        lines = text2.splitlines(True)
        i = 0
        while i < len(lines) and lines[i].startswith("#"):
            i += 1
        lines.insert(i, stamp)
        text2 = "".join(lines)
    REWARDS.write_text(text2, encoding="utf-8")
    return True, f"{key}: {cur:.4f} → {new:.4f}"


def accept_skill(bullet: str) -> None:
    REJECTED.mkdir(parents=True, exist_ok=True)
    BEST.parent.mkdir(parents=True, exist_ok=True)
    text = SKILL.read_text(encoding="utf-8")
    marker = "CHANGE LOG:\n"
    line = f"{bullet} — WHY: self_heal_epoch accept"
    if marker in text:
        text = text.replace(marker, marker + f"- {line}\n", 1)
    else:
        text = f"CHANGE LOG:\n- {line}\n\n" + text
    section = "## Known failure modes (from telemetry)\n"
    if section in text:
        at = text.index(section) + len(section)
        text = text[:at] + f"\n- {bullet}\n" + text[at:]
    else:
        text += f"\n{section}\n- {bullet}\n"
    SKILL.write_text(text, encoding="utf-8")
    shutil.copy2(SKILL, BEST)


def reject_skill(bullet: str) -> Path:
    REJECTED.mkdir(parents=True, exist_ok=True)
    path = REJECTED / f"{date.today()}_{len(list(REJECTED.glob('*')))}.txt"
    path.write_text(bullet + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Self-heal epoch")
    ap.add_argument("brain", nargs="?", default="PROVEN_SPRINT_row04_clear24_2026-07-20")
    ap.add_argument("goal", nargs="?", type=float, default=3.0)
    ap.add_argument("floor", nargs="?", type=float, default=3.5)
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--sprint-minutes", type=float, default=0.0)
    ap.add_argument("--auto-accept-skill", action="store_true")
    ap.add_argument("--apply-reward-nudge", action="store_true")
    args = ap.parse_args()

    EPOCH_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    epoch_path = EPOCH_DIR / f"epoch_{stamp}.json"

    print("=" * 64)
    print("SELF-HEAL EPOCH — self-correct + self-improve")
    print(f"brain={args.brain}  goal={args.goal}%  floor={args.floor}%")
    print("=" * 64)

    print("\n[0] Baseline prove_it...")
    r0 = run([sys.executable, "scripts/prove_it.py", args.brain, str(args.goal), str(args.floor)])
    print(r0.stdout)
    base = parse_prove_it(r0.stdout)
    print(f"    baseline clear={base['clear_pct']}% breach={base['breach_pct']}% row={base['row']}")

    print("\n[1] Trajectories (Mind + Ghosts + IRAC)...")
    # Mind Probe over many days is slow on CPU — allow ~3 min/day + buffer
    _traj_timeout = max(1800, int(args.days) * 180 + 600)
    r1 = run([
        sys.executable, "scripts/give_llm_what_it_needs.py",
        args.brain, str(args.goal), str(args.floor),
        "--days", str(args.days),
    ], timeout=_traj_timeout)
    print(r1.stdout[-3000:] if r1.stdout else "")

    irac = load_irac(args.brain)
    if not irac:
        print("FAIL: no IRAC JSON")
        return 1
    proposal = propose_from_irac(irac)
    print("    proposal class:", proposal["class"])
    print("    skill:", proposal["skill_bullet"])
    print("    reward_nudge:", proposal["reward_nudge"])

    if args.sprint_minutes > 0:
        print(f"\n[2] Frontier consistency_sprint ({args.sprint_minutes} min)...")
        r2 = run([
            sys.executable, "scripts/consistency_sprint.py",
            "--minutes", str(args.sprint_minutes), "--envs", "64",
        ], timeout=int(args.sprint_minutes * 60 + 300))
        print(r2.stdout[-2000:] if r2.stdout else "")
    else:
        print("\n[2] Sprint skipped (pass --sprint-minutes N)")

    print("\n[3] Post prove_it gate...")
    r3 = run([sys.executable, "scripts/prove_it.py", args.brain, str(args.goal), str(args.floor)])
    print(r3.stdout)
    post = parse_prove_it(r3.stdout)

    gate_ok = True
    reasons = []
    if post["breach_pct"] is not None and post["breach_pct"] > 0:
        gate_ok = False
        reasons.append(f"breach {post['breach_pct']}% > 0")
    if (
        base["clear_pct"] is not None
        and post["clear_pct"] is not None
        and post["clear_pct"] + 1e-6 < base["clear_pct"]
        and args.sprint_minutes > 0
    ):
        gate_ok = False
        reasons.append(f"clear {post['clear_pct']}% < baseline {base['clear_pct']}%")

    accepted = False
    reward_msg = None
    if gate_ok and args.auto_accept_skill:
        accept_skill(proposal["skill_bullet"])
        accepted = True
        print("\n[4] SKILL ACCEPTED → doctrine/policy_skill.md")
        if args.apply_reward_nudge:
            msgs = []
            nudges = proposal.get("reward_nudges") or (
                [proposal["reward_nudge"]] if proposal.get("reward_nudge") else []
            )
            for nd in nudges:
                if not nd:
                    continue
                ok, msg = apply_reward_nudge(nd["key"], nd["delta"])
                msgs.append(msg if ok else f"skip ({msg})")
            reward_msg = "; ".join(msgs) if msgs else None
            print("    reward:", reward_msg or "no nudge")
    elif gate_ok:
        path = reject_skill(proposal["skill_bullet"] + " [pending --auto-accept-skill]")
        print(f"\n[4] Gate OK; skill pending. Review then --auto-accept-skill. {path}")
    else:
        path = reject_skill(proposal["skill_bullet"] + " | gate_fail: " + "; ".join(reasons))
        print(f"\n[4] REJECTED — {reasons} → {path}")

    record = {
        "stamp": stamp,
        "brain": args.brain,
        "goal": args.goal,
        "floor": args.floor,
        "baseline": {k: base[k] for k in ("clear_pct", "breach_pct", "row")},
        "post": {k: post[k] for k in ("clear_pct", "breach_pct", "row")},
        "proposal": proposal,
        "gate_ok": gate_ok,
        "gate_reasons": reasons,
        "skill_accepted": accepted,
        "reward_msg": reward_msg,
    }
    epoch_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\n[5] Epoch record → {epoch_path}")
    print("=" * 64)
    print("EPOCH COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
