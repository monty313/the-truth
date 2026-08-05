"""Run multi-pair tutor pre-training gate. Does NOT train, search dials, or promote.

Usage (repo root, PYTHONPATH=.;code):
  python lineages/adaptive_rl_brain_7_31_26/honest_gate/run_gate.py

Writes under lineages/adaptive_rl_brain_7_31_26/checkpoints/honest_gate/
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_LINEAGE = _HERE.parent
_ROOT = _LINEAGE.parent.parent
_CODE = _ROOT / "code"
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.honest_gate.bar_export import (
    export_day_bars,
    schema_column_names,
)
from lineages.adaptive_rl_brain_7_31_26.honest_gate.data_contract import (
    assert_no_day_leak,
    write_data_contract,
)
from lineages.adaptive_rl_brain_7_31_26.honest_gate.hashes import file_sha256, pin_paths
from lineages.adaptive_rl_brain_7_31_26.honest_gate.meaning_manifest import (
    assert_meaning_matches_frozen,
    build_meaning_manifest,
    write_frozen_manifest,
)
from lineages.adaptive_rl_brain_7_31_26.honest_gate.regime_report import (
    compare_practice_forward,
    write_regime_report,
)
from lineages.adaptive_rl_brain_7_31_26.honest_gate.score_schema import (
    build_conclusion_artifact,
    window_pass_rules,
    write_conclusion,
)
from lineages.adaptive_rl_brain_7_31_26.honest_gate.shell_lock import (
    ALLOWED_TRAINING_PARAMETERS,
    FORBIDDEN_TRAINING_PARAMETERS,
    assert_shell_locked,
    write_banned_families,
)
from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay, load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.score_ten_pairs import load_pairs_config

OUT_DIR = _LINEAGE / "checkpoints" / "honest_gate"
EXPERIMENT_ID = "multi_pair_tutor_honest_v1_pretrain_gate"
SEED = 42
DECODE = "heuristic"

CKPT = _LINEAGE / "checkpoints" / "multi_pair_consistent_v1.pt"
DIALS = _LINEAGE / "checkpoints" / "multi_pair_dials.json"


def _write_experiment_contract_md(
    path: Path,
    *,
    pins: Dict[str, Any],
    contract: Dict[str, Any],
    meaning: Dict[str, Any],
    shell: Dict[str, Any],
    gate_verdict: str,
) -> None:
    lines = [
        "# Multi-pair tutor — experiment contract (pre-training gate)",
        "",
        f"**Experiment ID:** `{EXPERIMENT_ID}`  ",
        f"**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}  ",
        f"**Gate verdict:** **{gate_verdict}**  ",
        f"**Track:** multi_pair_tutor (NOT PROVEN, NOT Channel1 claim)",
        "",
        "## Mission",
        "",
        "- Runtime **target%** / **risk%** without retrain.",
        "- **Clear** = banked equity% ≥ target% after costs path AND floor never hit.",
        "- **Breach** = floor touched → day fails.",
        "- Official meters: clear count, breach count, clear streak.",
        "- PnL / entries / “looks good” = diagnostics only.",
        "",
        "## Identity pins",
        "",
        f"| Item | Path | SHA-256 |",
        f"|------|------|---------|",
    ]
    for name, info in pins.items():
        h = info.get("sha256") or "MISSING"
        lines.append(f"| {name} | `{info.get('path')}` | `{h}` |")
    lines += [
        "",
        f"| decode | (claim path) | **{DECODE}** |",
        f"| seed | | **{SEED}** |",
        f"| meaning_version | | `{meaning.get('meaning_version')}` |",
        f"| meaning_hash | | `{meaning.get('meaning_hash')}` |",
        "",
        "## Data split (chronological)",
        "",
        f"- Source: `{contract['data_source']}` sha256=`{contract['data_sha256']}`",
        f"- Eligible days (min_bars≥{contract['min_bars_eligible']}): **{contract['n_eligible_days']}**",
        f"- Practice: **{contract['practice_day_count']}** days "
        f"({contract['practice_first']} → {contract['practice_last']})",
        f"- Forward: **{contract['forward_day_count']}** days "
        f"({contract['forward_first']} → {contract['forward_last']})",
        f"- Overlap: **{contract['leak_check']['overlap_n']}** (must be 0)",
        f"- 100-day conclusion: **{contract['hundred_day_conclusion']['status']}**",
        "",
        "## Shell",
        "",
        f"- SHELL_LOCKED: **{shell.get('SHELL_LOCKED')}** ok={shell.get('ok')}",
        f"- Laws: {', '.join(shell.get('shell_laws') or [])}",
        "",
        "## Prior claim honesty",
        "",
        "- `ten_pair_score_all.json` = **IN_SAMPLE_CLAIM** (dials may have seen all days historically).",
        "- Do **not** call prior forward JSON pure unseen if dials were fit with all-day search.",
        "- New training: dial search **practice only**; score forward once after freeze.",
        "",
        "## Allowed vs forbidden (first training cycle)",
        "",
        "**Allowed on practice only:** " + ", ".join(ALLOWED_TRAINING_PARAMETERS),
        "",
        "**Forbidden:** " + ", ".join(FORBIDDEN_TRAINING_PARAMETERS),
        "",
        "## Commands to reproduce this gate",
        "",
        "```powershell",
        '$env:PYTHONPATH = ".;code"',
        "python lineages/adaptive_rl_brain_7_31_26/honest_gate/run_gate.py",
        "```",
        "",
        "## What we do NOT claim yet",
        "",
        "- Not “consistent,” “unseen-proven,” “100-day,” “robust,” or “ready to train for production.”",
        "- Gate PASS means honesty infrastructure is ready — not that clear% is high.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _smoke_bar_export(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Export a few bars from one practice + one forward day; prove same schema."""
    days = load_calendar_days(contract["data_source"])
    p_dates = set(contract["practice_dates"])
    f_dates = set(contract["forward_dates"])
    practice_day = next((d for d in days if str(d[0]) in p_dates), None)
    forward_day = next((d for d in days if str(d[0]) in f_dates), None)
    if practice_day is None or forward_day is None:
        return {"ok": False, "error": "missing practice or forward day"}
    dials = json.loads(DIALS.read_text(encoding="utf-8")).get("dials", {})
    common = dict(
        target_pct=1.0,
        risk_pct=2.0,
        risk_use_frac=float(dials.get("risk_use_frac", 0.35)),
        stop_atr_mult=float(dials.get("stop_atr_mult", 2.0)),
        per_trade_cap_pct=float(dials.get("per_trade_cap_pct", 0.25)),
    )
    pd_obj = GoalEquityDay(practice_day[1], date_str=str(practice_day[0]), **common)
    fd_obj = GoalEquityDay(forward_day[1], date_str=str(forward_day[0]), **common)
    p_rows = export_day_bars(pd_obj, split="practice", max_decisions=3)
    f_rows = export_day_bars(fd_obj, split="forward", max_decisions=3)
    cols = schema_column_names()
    ok = True
    errors: List[str] = []
    if p_rows:
        if list(p_rows[0].keys()) != cols:
            # keys may be same set
            if set(p_rows[0].keys()) != set(cols):
                ok = False
                errors.append("practice columns mismatch schema")
    if f_rows:
        if set(f_rows[0].keys()) != set(cols):
            ok = False
            errors.append("forward columns mismatch schema")
    if p_rows and f_rows and set(p_rows[0].keys()) != set(f_rows[0].keys()):
        ok = False
        errors.append("practice vs forward column set differs")
    out = {
        "ok": ok and bool(p_rows) and bool(f_rows),
        "schema_columns": cols,
        "practice_date": str(practice_day[0]),
        "forward_date": str(forward_day[0]),
        "practice_rows_exported": len(p_rows),
        "forward_rows_exported": len(f_rows),
        "errors": errors,
    }
    sample_path = OUT_DIR / "bar_export_sample.json"
    sample_path.write_text(
        json.dumps(
            {"practice_sample": p_rows[:2], "forward_sample": f_rows[:2], "meta": out},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def _demo_regime_report() -> Dict[str, Any]:
    """Minimal synthetic tag table to prove insufficient-evidence rule (not market truth)."""
    practice = (
        [{"tag": "higher_bull", "cleared": True, "breached": False} for _ in range(12)]
        + [{"tag": "higher_bull", "cleared": False, "breached": False} for _ in range(4)]
        + [{"tag": "rare_tag", "cleared": True, "breached": False} for _ in range(2)]
    )
    forward = (
        [{"tag": "higher_bull", "cleared": True, "breached": False} for _ in range(6)]
        + [{"tag": "higher_bull", "cleared": False, "breached": False} for _ in range(10)]
        + [{"tag": "rare_tag", "cleared": False, "breached": False} for _ in range(1)]
    )
    rep = compare_practice_forward(practice, forward, min_samples=8)
    write_regime_report(OUT_DIR / "regime_report_demo.json", rep)
    return rep


def main(argv: Optional[List[str]] = None) -> int:
    del argv  # unused; gate has no train flags
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures: List[str] = []
    checks: Dict[str, Any] = {}

    # 1) Meaning freeze + match
    meaning = write_frozen_manifest(OUT_DIR / "meaning_manifest.json")
    try:
        mcheck = assert_meaning_matches_frozen(OUT_DIR / "meaning_manifest.json")
        checks["meaning"] = mcheck
    except Exception as e:
        failures.append(f"meaning: {e}")
        checks["meaning"] = {"ok": False, "error": str(e)}

    # 2) Shell lock + ban list
    write_banned_families(OUT_DIR / "banned_rule_families.json")
    try:
        shell = assert_shell_locked()
        checks["shell"] = shell
    except Exception as e:
        failures.append(f"shell: {e}")
        checks["shell"] = {"ok": False, "error": str(e)}

    # 3) Data contract + leak
    try:
        contract = write_data_contract(OUT_DIR / "data_contract.json")
        leak = assert_no_day_leak(contract["practice_dates"], contract["forward_dates"])
        checks["data_contract"] = {
            "ok": True,
            "n_eligible": contract["n_eligible_days"],
            "practice_n": contract["practice_day_count"],
            "forward_n": contract["forward_day_count"],
            "leak": leak,
            "hundred_day": contract["hundred_day_conclusion"],
        }
        # also write date lists alone for deliverable B
        (OUT_DIR / "practice_dates.json").write_text(
            json.dumps(contract["practice_dates"], indent=2) + "\n", encoding="utf-8"
        )
        (OUT_DIR / "forward_dates.json").write_text(
            json.dumps(contract["forward_dates"], indent=2) + "\n", encoding="utf-8"
        )
    except Exception as e:
        failures.append(f"data_contract: {e}")
        checks["data_contract"] = {"ok": False, "error": str(e)}
        contract = {}

    # 4) Identity pins
    cfg = load_pairs_config()
    data_name = Path(cfg.get("data_source", "XAUUSD_curriculum_2026.csv")).name
    from lineages.adaptive_rl_brain_7_31_26.price_data import resolve_raw_csv

    data_path = resolve_raw_csv(data_name)
    pins = pin_paths(
        {
            "checkpoint": CKPT,
            "dials": DIALS,
            "data": data_path,
            "meaning_manifest": OUT_DIR / "meaning_manifest.json",
            "banned_rule_families": OUT_DIR / "banned_rule_families.json",
            "data_contract": OUT_DIR / "data_contract.json",
        }
    )
    for req in ("checkpoint", "dials", "data"):
        if not pins[req]["exists"]:
            failures.append(f"missing pin file: {req}")
    checks["pins"] = pins
    checks["decode"] = DECODE
    checks["seed"] = SEED
    checks["experiment_id"] = EXPERIMENT_ID
    checks["tracks_separated"] = {
        "multi_pair_tutor": True,
        "PROVEN_champion": "never auto-write models/PROVEN_*.pt",
        "channel1_rl": "separate scoreboard; not claim decode",
    }

    # 5) Dial-search policy statement (code path check)
    train_src = (_LINEAGE / "train_multi_pair.py").read_text(encoding="utf-8")
    practice_only = (
        "practice only" in train_src.lower()
        or "search_dials(practice" in train_src
        or "SEARCH_WINDOW = \"practice\"" in train_src
        or 'search_window": "practice"' in train_src
        or "on practice" in train_src.lower()
    )
    # After our edit we require explicit practice-only path
    has_practice_search = "search_dials(practice" in train_src or "search_days = practice" in train_src
    has_all_days_default = "Search on ALL days" in train_src and "search_dials(all_days" in train_src
    if has_all_days_default and not has_practice_search:
        failures.append(
            "train_multi_pair.py still defaults dial search to ALL days — must be practice-only"
        )
        checks["dial_search_policy"] = {"ok": False, "contaminated_default": True}
    else:
        checks["dial_search_policy"] = {
            "ok": True,
            "practice_only_path_present": has_practice_search or practice_only,
            "note": "Prior all-day results remain IN_SAMPLE_CLAIM",
        }

    # 6) Bar export smoke + regime demo
    if contract:
        try:
            checks["bar_export"] = _smoke_bar_export(contract)
            if not checks["bar_export"].get("ok"):
                failures.append("bar_export smoke failed: " + str(checks["bar_export"]))
        except Exception as e:
            failures.append(f"bar_export: {e}")
            checks["bar_export"] = {"ok": False, "error": str(e)}
    try:
        checks["regime_demo"] = _demo_regime_report()
        # ensure small sample not called lying
        rare = [
            c
            for c in checks["regime_demo"]["comparisons"]
            if c.get("tag") == "rare_tag"
        ]
        if rare and rare[0].get("status") != "INSUFFICIENT_EVIDENCE":
            failures.append("regime report must mark rare_tag insufficient, not lying")
        if any(c.get("shell_change_authorized") for c in checks["regime_demo"]["comparisons"]):
            failures.append("regime report must not authorize shell changes")
    except Exception as e:
        failures.append(f"regime: {e}")
        checks["regime_demo"] = {"ok": False, "error": str(e)}

    # 7) Score rules freeze
    rules = window_pass_rules()
    (OUT_DIR / "score_rules.json").write_text(
        json.dumps(rules, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checks["score_rules"] = {"ok": True, "path": str(OUT_DIR / "score_rules.json")}

    gate_ok = len(failures) == 0
    hundred = (contract or {}).get("hundred_day_conclusion", {})
    if hundred.get("status") == "NOT_YET_MEASURABLE":
        # Not a gate failure — honesty status for 100-day claims
        checks["hundred_day_note"] = hundred

    verdict = "GATE_PASS" if gate_ok else "GATE_FAIL"
    # Training readiness vs 100-day claim
    cannot_claim = [
        "100-day consistency (eligible days < 100)"
        if hundred.get("status") == "NOT_YET_MEASURABLE"
        else None,
        "pure unseen multi-pair performance from prior ten_pair_score_* (IN_SAMPLE_CLAIM / possible CONTAMINATED dial fit)",
        "robust / ready for production",
        "PROVEN champion replacement",
        "any KEEP on a new training candidate (no training run yet)",
    ]
    cannot_claim = [c for c in cannot_claim if c]

    conclusion = build_conclusion_artifact(
        verdict=verdict,
        reason=(
            "Pre-training honesty gate complete; infrastructure ready."
            if gate_ok
            else "Gate failures: " + "; ".join(failures)
        ),
        experiment_id=EXPERIMENT_ID,
        pins={
            "checkpoint_sha256": pins.get("checkpoint", {}).get("sha256"),
            "dials_sha256": pins.get("dials", {}).get("sha256"),
            "data_sha256": pins.get("data", {}).get("sha256"),
            "meaning_hash": meaning.get("meaning_hash"),
            "decode": DECODE,
            "seed": SEED,
        },
        windows={
            "practice_dates": contract.get("practice_dates"),
            "forward_dates": contract.get("forward_dates"),
            "n_eligible": contract.get("n_eligible_days"),
            "hundred_day": hundred,
        },
        extra={
            "failures": failures,
            "checks_summary": {k: (v.get("ok") if isinstance(v, dict) and "ok" in v else True) for k, v in checks.items() if k != "pins"},
            "cannot_yet_claim": cannot_claim,
            "prior_claim_label": "IN_SAMPLE_CLAIM",
            "allowed_training_parameters": list(ALLOWED_TRAINING_PARAMETERS),
            "forbidden_training_parameters": list(FORBIDDEN_TRAINING_PARAMETERS),
            "training_started": False,
            "dials_tuned_this_run": False,
            "promoted": False,
        },
    )
    write_conclusion(OUT_DIR / "last_score_verdict.json", conclusion)

    report = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "failures": failures,
        "checks": checks,
        "cannot_yet_claim": cannot_claim,
        "score_rules": rules,
        "outputs_dir": str(OUT_DIR).replace("\\", "/"),
    }
    (OUT_DIR / "PRETRAIN_GATE_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    shell_for_md = checks.get("shell") or {"SHELL_LOCKED": True, "ok": False, "shell_laws": []}
    _write_experiment_contract_md(
        OUT_DIR / "EXPERIMENT_CONTRACT.md",
        pins=pins,
        contract=contract or {},
        meaning=meaning,
        shell=shell_for_md,
        gate_verdict=verdict,
    )

    # Human one-page also as GATE_REPORT.md
    md_lines = [
        f"# PRE-TRAINING GATE REPORT — {verdict}",
        "",
        f"Experiment: `{EXPERIMENT_ID}`  ",
        f"UTC: {report['generated_at_utc']}",
        "",
        "## Failures",
        "",
    ]
    if failures:
        for f in failures:
            md_lines.append(f"- FAIL: {f}")
    else:
        md_lines.append("- none")
    md_lines += [
        "",
        "## Hashes",
        "",
        f"- checkpoint: `{pins.get('checkpoint', {}).get('sha256')}`",
        f"- dials: `{pins.get('dials', {}).get('sha256')}`",
        f"- data: `{pins.get('data', {}).get('sha256')}`",
        f"- meaning: `{meaning.get('meaning_hash')}`",
        f"- decode: `{DECODE}` seed: `{SEED}`",
        "",
        "## Split",
        "",
        f"- practice n={contract.get('practice_day_count')} "
        f"{contract.get('practice_first')}→{contract.get('practice_last')}",
        f"- forward n={contract.get('forward_day_count')} "
        f"{contract.get('forward_first')}→{contract.get('forward_last')}",
        f"- overlap=0 enforced",
        f"- 100-day: {hundred.get('status')}",
        "",
        "## Shell / leak",
        "",
        f"- shell ok: {shell_for_md.get('ok')}",
        f"- leak ok: {(contract or {}).get('leak_check', {}).get('ok')}",
        "",
        "## Cannot yet claim",
        "",
    ]
    for c in cannot_claim:
        md_lines.append(f"- {c}")
    md_lines += [
        "",
        "## Artifacts",
        "",
        f"- `{OUT_DIR.as_posix()}/EXPERIMENT_CONTRACT.md`",
        f"- `{OUT_DIR.as_posix()}/PRETRAIN_GATE_REPORT.json`",
        f"- `{OUT_DIR.as_posix()}/last_score_verdict.json`",
        f"- `{OUT_DIR.as_posix()}/data_contract.json`",
        f"- `{OUT_DIR.as_posix()}/meaning_manifest.json`",
        "",
        "**No training, dial search, reward change, or promotion was performed.**",
        "",
    ]
    (OUT_DIR / "PRETRAIN_GATE_REPORT.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("=" * 64)
    print(f"MULTI-PAIR TUTOR PRE-TRAINING GATE — {verdict}")
    print("=" * 64)
    print("checkpoint", pins.get("checkpoint", {}).get("sha256"))
    print("dials     ", pins.get("dials", {}).get("sha256"))
    print("data      ", pins.get("data", {}).get("sha256"))
    print("meaning   ", meaning.get("meaning_hash"))
    print("decode    ", DECODE, "seed", SEED)
    if contract:
        print(
            "days      ",
            contract.get("n_eligible_days"),
            "practice",
            contract.get("practice_day_count"),
            "forward",
            contract.get("forward_day_count"),
            "overlap",
            contract.get("leak_check", {}).get("overlap_n"),
        )
        print("100-day   ", hundred.get("status"))
    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
    print("artifacts ", OUT_DIR)
    print("training  NOT STARTED (by design)")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
