"""Config loader — THE single door to every number (LAWS #3 enforcement).

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty (audit 2026-07-19 finding: 6 of 8 config files
       were decorative — code hardcoded the numbers. That violated
       LAWS #3 "no hidden thresholds" and the promise that typing a
       change into configs/ changes the bot).
WHAT:  load(name) -> dict, cached; every module gets its numbers HERE.
       Also exposes ROOT and path helpers so scripts work from any cwd
       (audit: cwd-relative paths broke everything outside repo root).
WHEN:  2026-07-19 (post-audit rebuild).
WHERE: imported by simulator, env, ppo, rewards, engine, bridge, HUD,
       and every script.
WHY:   One door means the meta-optimizer's approved proposals and
       Monty's typed X actually reach the machine.
INTERCONNECTED WITH: configs/*.yaml (in), everything else (out),
       tests/test_configs.py (proof the door is real).
----------------------------------------------------------------------

CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  created  — WHY: 6/8 config files were decorative; one door so typing a config changes the machine (audit S7/R11).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
from __future__ import annotations
import os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE: dict = {}


def path(*parts: str) -> str:
    """Absolute path inside the repo — scripts must never use cwd-relative."""
    return os.path.join(ROOT, *parts)


def load(name: str, refresh: bool = False) -> dict:
    """Load configs/<name>.yaml (cached). refresh=True re-reads from disk."""
    if refresh or name not in _CACHE:
        with open(path("configs", f"{name}.yaml"), encoding="utf-8") as f:
            _CACHE[name] = yaml.safe_load(f) or {}
    return _CACHE[name]


def shell_cfg() -> dict:
    """The Shell's numbers, from masks_shell.yaml + data.yaml symbol info."""
    c = dict(load("masks_shell"))
    d = load("data")
    c["point_size"] = d.get("point_size", 0.01)
    c["contract_size"] = d.get("contract_size", 100.0)
    return c


def goals_cfg() -> dict:
    return load("goals")


def training_cfg() -> dict:
    return load("training")


def policy_hidden() -> int:
    """ONE door for the brain's hidden size (review 2026-07-20: it was hardcoded as
    128 in six call sites; a config change would have silently built mismatched brains)."""
    return int(training_cfg().get("policy", {}).get("hidden", 128))


def decide_every() -> int:
    """ONE door for the decision cadence (act once per N bars, hold between). Semantics-
    bearing: training and deployment must share it, so it lives in training.yaml."""
    return int(training_cfg().get("decide_every", 5))
