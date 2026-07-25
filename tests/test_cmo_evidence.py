"""CMO diagnostic must be evidence-based and consistency-oriented."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_she():
    spec = importlib.util.spec_from_file_location(
        "self_heal_epoch", ROOT / "scripts" / "self_heal_epoch.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_doctrine_core_rules():
    doc = (ROOT / "doctrine" / "SYSTEM_DOCTRINE_CMO.md").read_text()
    assert "prove_it" in doc
    assert "impossible" in doc.lower() and "forbidden" in doc.lower()
    assert "last resort" in doc.lower()


def test_proposal_requires_evidence():
    she = _load_she()
    weak = {
        "application": {"sum_policy_hold_on_setup": 0, "sum_high_miss_pull": 0},
        "conclusion": {"class": "Policy"},
        "issue": "none",
    }
    assert she.propose_from_irac(weak)["reward_nudge"] is None

    strong = {
        "application": {"sum_policy_hold_on_setup": 200, "sum_high_miss_pull": 20},
        "conclusion": {"class": "Policy"},
        "issue": "hold",
    }
    p = she.propose_from_irac(strong)
    assert p["reward_nudge"] is not None
    assert p["reward_nudge"]["key"] == "w_pullback_with_htf"
    assert "policy_hold" in p["skill_bullet"]


def test_gate_rejects_breach():
    she = _load_she()
    bad = she.parse_prove_it(
        "cleared (hit target, NO breach): 40% of days\n"
        "breached the risk floor: 2% of days\n"
    )
    assert bad["breach_pct"] == 2.0
    assert bad["clear_pct"] == 40.0
    assert bad["breach_pct"] > 0


def test_no_retail_vagueness_in_strong_proposal():
    she = _load_she()
    p = she.propose_from_irac({
        "application": {"sum_policy_hold_on_setup": 100, "sum_high_miss_pull": 15},
        "conclusion": {"class": "Policy"},
        "issue": "x",
    })
    b = p["skill_bullet"].lower()
    assert "overbought" not in b
    assert "rsi crosses" not in b
