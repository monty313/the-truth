#!/usr/bin/env python3
"""SkillOpt-style gate for doctrine/policy_skill.md

Memory loop for the diagnostic LLM:
  trajectories (IRAC JSON) → proposed skill bullet → validate with prove_it philosophy
  → accept to policy_skill.md + best_skill.md OR reject to artifacts/skills/rejected/

Usage:
  python scripts/skillopt_gate.py --from-irac artifacts/llm_curriculum/irac_BRAIN.json --accept
  python scripts/skillopt_gate.py --add "evidence line" --accept
  python scripts/skillopt_gate.py --snapshot-best

Only --accept when prove_it improved or IRAC evidence is hard.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL = os.path.join(ROOT, "doctrine", "policy_skill.md")
BEST = os.path.join(ROOT, "artifacts", "skills", "best_skill.md")
REJECTED = os.path.join(ROOT, "artifacts", "skills", "rejected")


def ensure_dirs():
    os.makedirs(os.path.dirname(BEST), exist_ok=True)
    os.makedirs(REJECTED, exist_ok=True)


def load_skill() -> str:
    with open(SKILL, encoding="utf-8") as f:
        return f.read()


def append_changelog(text: str, line: str) -> str:
    marker = "CHANGE LOG:\n"
    if marker not in text:
        return f"CHANGE LOG:\n- {line}\n\n" + text
    return text.replace(marker, marker + f"- {line}\n", 1)


def append_failure_mode(text: str, bullet: str) -> str:
    section = "## Known failure modes (from telemetry)\n"
    if section not in text:
        return text + f"\n{section}\n- {bullet}\n"
    insert_at = text.index(section) + len(section)
    return text[:insert_at] + f"\n- {bullet}\n" + text[insert_at:]


def from_irac(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    app = d.get("application", {})
    conc = d.get("conclusion", {})
    return (
        f"{date.today()}: IRAC class={conc.get('class', 'Policy')}; "
        f"policy_hold_on_setup={app.get('sum_policy_hold_on_setup', '?')}; "
        f"high_miss_pull={app.get('sum_high_miss_pull', '?')}; "
        f"cure={conc.get('cure_in_config', '')}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-irac")
    ap.add_argument("--add")
    ap.add_argument("--accept", action="store_true")
    ap.add_argument("--reject", action="store_true")
    ap.add_argument("--snapshot-best", action="store_true")
    args = ap.parse_args()
    ensure_dirs()

    if args.snapshot_best:
        shutil.copy2(SKILL, BEST)
        print(f"snapshot → {BEST}")
        return 0

    if args.from_irac:
        bullet = from_irac(args.from_irac)
    elif args.add:
        bullet = args.add
    else:
        ap.print_help()
        return 2

    if args.reject or not args.accept:
        out = os.path.join(REJECTED, f"{date.today()}_{len(os.listdir(REJECTED))}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(bullet + "\n")
        print(f"REJECTED: {out}")
        print(bullet)
        if not args.accept and not args.reject:
            print("Hint: pass --accept only when prove_it improved or evidence is hard.")
        return 0

    text = load_skill()
    text = append_changelog(text, f"{bullet} — WHY: SkillOpt gated accept")
    text = append_failure_mode(text, bullet)
    with open(SKILL, "w", encoding="utf-8") as f:
        f.write(text)
    shutil.copy2(SKILL, BEST)
    print(f"ACCEPTED → {SKILL}")
    print(bullet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
