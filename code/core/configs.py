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
- 2026-07-31  FinRL-clean layout: packages under code/; artifacts/reports/logs
  under outputs/ — WHY: fewer root folders; path() aliases keep old call sites working.
- 2026-07-30  ROOT auto-detect: the-truth/core vs the-truth/src/core — WHY: cookiecutter move broke configs when package lives at repo root.
- 2026-07-30  ROOT under src/ + data_file/models_dir/checkpoint_file  — WHY: cookiecutter-mlops tidy.
- 2026-07-19  created  — WHY: 6/8 config files were decorative; one door so typing a config changes the machine (audit S7/R11).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
from __future__ import annotations
import os
import yaml

def _find_root() -> str:
    """Repo root: directory that contains configs/ (+ scripts or data)."""
    here = os.path.abspath(__file__)
    d = os.path.dirname(here)
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "configs")) and (
            os.path.isdir(os.path.join(d, "scripts"))
            or os.path.isdir(os.path.join(d, "data"))
            or os.path.isdir(os.path.join(d, "code"))
        ):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # code/core -> repo root is two levels up from core/
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = _find_root()
_CACHE: dict = {}

# Old top-level names → new tidy homes (FinRL-clean layout 2026-07-31)
_PATH_ALIASES = {
    "artifacts": ("outputs", "artifacts"),
    "reports": ("outputs", "reports"),
    "logs": ("outputs", "logs"),
}


def path(*parts: str) -> str:
    """Absolute path inside the repo — scripts must never use cwd-relative."""
    if parts and parts[0] in _PATH_ALIASES:
        parts = _PATH_ALIASES[parts[0]] + tuple(parts[1:])
    return os.path.join(ROOT, *parts)


def data_file(*parts: str) -> str:
    """Price/feature CSVs live in data/raw/ (cookiecutter). Falls back to data/."""
    preferred = path("data", "raw", *parts)
    if os.path.exists(preferred):
        return preferred
    legacy = path("data", *parts)
    return preferred if not os.path.exists(legacy) else legacy


def models_dir() -> str:
    """Trained brains live in models/ (cookiecutter)."""
    d = path("models")
    os.makedirs(d, exist_ok=True)
    os.makedirs(path("models", "history"), exist_ok=True)
    return d


def checkpoint_file(name: str) -> str:
    """Resolve models/<name>.pt with legacy artifacts/checkpoints/ fallback."""
    if not name.endswith(".pt"):
        name = f"{name}.pt"
    preferred = path("models", name)
    if os.path.exists(preferred):
        return preferred
    legacy = path("artifacts", "checkpoints", name)
    return preferred if not os.path.exists(legacy) else legacy


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
