"""SHELL_LOCKED laws + banned rule families for multi-pair tutor claim path.

CHANGE LOG:
- 2026-07-31  honest gate — WHY: IRAC-01 trail/cushion/scale-in package destroyed
  multi-pair (6/10 → 0/10). Shell must not move during attention tuning.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

_LINEAGE = Path(__file__).resolve().parents[1]
EQUITY_DAY = _LINEAGE / "equity_day.py"
BANNED_PATH = _LINEAGE / "checkpoints" / "honest_gate" / "banned_rule_families.json"

# Immutable during attention/policy tuning (multi-pair tutor claim path).
SHELL_LOCKED = True

SHELL_LAWS: Tuple[str, ...] = (
    "heat_refuse_open",  # residual floor heat; refuse when ~0
    "floor_scaled_sizing",  # size scales with runtime risk%
    "every_bar_marks",  # stop/breach/bank between decision bars
    "bank_at_target",  # flatten when equity% >= target%
    "breach_termination",  # floor touch → dead day; never clear
    "one_signal_flat_and_in_trade",  # same eyes; reverse only on opposite
)

# Attention dials that MAY be searched on practice only (not shell physics).
ALLOWED_ATTENTION_DIALS: Tuple[str, ...] = (
    "risk_use_frac",
    "stop_atr_mult",
    "per_trade_cap_pct",
)

# Forbidden to smuggle in as “optimization.”
BANNED_RULE_FAMILIES: Tuple[Dict[str, Any], ...] = (
    {
        "id": "R1",
        "name": "trail_cushion_scale_in_package",
        "why": "IRAC-01: multi-pair pass 6/10 → 0/10",
        "patterns": [
            r"trail_stop",
            r"trailing_stop",
            r"floor_cushion",
            r"big_cushion",
            r"scale_in.*trail",
            r"trail.*scale",
        ],
    },
    {
        "id": "R2",
        "name": "decision_bar_only_stops",
        "why": "Floor walk-through between strides; dishonest breach",
        "require_positive": [r"_mark_bar", r"for bt in range"],
    },
    {
        "id": "R3",
        "name": "weaker_in_trade_signal",
        "why": "High targets stuck; IRAC-03 kept one signal path",
        "patterns": [],  # enforced by code inspection of recommended_action
    },
    {
        "id": "R4",
        "name": "pure_greedy_as_default_claim",
        "why": "Channel1 pure greedy freezes; claim decode is heuristic",
        "note": "decode must be heuristic for multi-pair claim path",
    },
)

FORBIDDEN_TRAINING_PARAMETERS: Tuple[str, ...] = (
    "target_pct_baked_into_weights",
    "risk_pct_baked_into_weights",
    "trail_stop_enabled",
    "floor_cushion_pct",
    "scale_in_on_claim_path",
    "decision_bar_only_stops",
    "weaker_in_trade_eyes",
    "decode=pure_greedy_as_claim_default",
    "forward_day_in_dial_search",
    "forward_day_in_reward_selection",
    "forward_day_in_feature_selection",
    "shell_law_mutation_without_unlock",
    "retrain_only_because_target_risk_changed",
    "cross_track_promote_to_PROVEN",
    "cross_track_promote_channel1_to_multi_pair_claim",
)

ALLOWED_TRAINING_PARAMETERS: Tuple[str, ...] = (
    "risk_use_frac",  # attention dial — practice only
    "stop_atr_mult",  # attention dial — practice only
    "per_trade_cap_pct",  # attention dial — practice only
    "channel1_policy_weights",  # BC/policy on practice only; claim decode stays heuristic unless separate exp
    "bc_epochs",
    "bc_lr",
    "hidden_size",
    "seed",
    "practice_day_subset_for_bc",
)


def write_banned_families(path: Path | str | None = None) -> Path:
    path = Path(path) if path is not None else BANNED_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "schema_version": 1,
        "SHELL_LOCKED": SHELL_LOCKED,
        "shell_laws": list(SHELL_LAWS),
        "allowed_attention_dials": list(ALLOWED_ATTENTION_DIALS),
        "banned_rule_families": list(BANNED_RULE_FAMILIES),
        "allowed_training_parameters": list(ALLOWED_TRAINING_PARAMETERS),
        "forbidden_training_parameters": list(FORBIDDEN_TRAINING_PARAMETERS),
        "unlock_requires": [
            "explicit SHELL_LOCKED=False flag in experiment identity",
            "new experiment_id",
            "full practice+forward+claim re-score",
            "never smuggle via dial grid",
        ],
    }
    path.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def verify_shell_locked(equity_day_path: Path | str | None = None) -> Dict[str, Any]:
    """Inspect equity_day.py for locked laws present and banned trail package absent.

    Returns {ok, errors, warnings, checks}.
    """
    path = Path(equity_day_path) if equity_day_path is not None else EQUITY_DAY
    text = path.read_text(encoding="utf-8")
    errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, bool] = {}

    # Required symbols / behaviors
    required = {
        "heat_refuse_open": r"risk_use_frac|risk_frac",
        "floor_scaled_sizing": r"floor_scale",
        "every_bar_marks": r"def _mark_bar",
        "bank_at_target": r"banked\s*=\s*True|self\.banked",
        "breach_termination": r"self\.breached\s*=\s*True",
        "one_signal_flat_and_in_trade": r"def recommended_action",
        "mark_loop_between_decisions": r"for bt in range\(prev_t",
        "force_flat_perception": r"self\.runner\.position\s*=\s*None",
    }
    for name, pat in required.items():
        ok = re.search(pat, text) is not None
        checks[name] = ok
        if not ok:
            errors.append(f"missing shell law signal: {name} (/{pat}/)")

    # Banned trail package strings in equity_day claim path body
    # (CHANGE LOG may mention REVERT — allow comment mentions of trail only in header)
    body = text
    # Strip module docstring for ban scan of active code
    if '"""' in body:
        parts = body.split('"""')
        if len(parts) >= 3:
            body_code = '"""'.join(parts[2:])
        else:
            body_code = body
    else:
        body_code = body

    banned_active = [
        (r"trail_stop\s*=", "trail_stop assignment"),
        (r"trailing_stop", "trailing_stop"),
        (r"floor_cushion", "floor_cushion"),
        (r"scale_in\s*=\s*True", "scale_in=True on claim path"),
    ]
    for pat, label in banned_active:
        if re.search(pat, body_code):
            errors.append(f"banned family present in equity_day body: {label}")
            checks[f"ban_{label}"] = False
        else:
            checks[f"ban_{label}"] = True

    # recommended_action must reverse only on opposite (not weaker in-trade branch)
    if "Force flat perception" not in text and "runner.position = None" not in text:
        errors.append("recommended_action may not force flat perception (R3 risk)")

    ok = len(errors) == 0
    return {
        "ok": ok,
        "SHELL_LOCKED": SHELL_LOCKED,
        "shell_laws": list(SHELL_LAWS),
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "path": str(path).replace("\\", "/"),
    }


def assert_shell_locked() -> Dict[str, Any]:
    result = verify_shell_locked()
    if not result["ok"]:
        raise AssertionError(
            "SHELL_LOCKED verification failed: " + "; ".join(result["errors"])
        )
    return result
