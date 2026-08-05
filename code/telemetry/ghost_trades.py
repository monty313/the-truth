"""Ghost Trades — counterfactual evidence for IRAC Application.

5W+I -----------------------------------------------------------------
WHO:   Fable 5 for Monty (Project Instructions Diagnostic LLM layer).
WHAT:  Given a mind-dump decision, compute what alternative ops would have
       aligned with Gravity patterns so IRAC Application can cite hard evidence.
WHEN:  2026-07-24 Phase 2 of autonomous self-heal plan.
WHERE: Called from scripts/diagnose_day.py after Mind Probe.
WHY:   "What if you had taken the pullback long instead of hold?" is the evidence
       the Diagnostic LLM needs. Constraint: read-only — no weight or obs change.
INTERCONNECTED WITH: telemetry/mind_probe.py, doctrine/STANDING_LAWS.md (IRAC),
       configs/rewards.yaml, training/fastsim (op encoding).
----------------------------------------------------------------------

CHANGE LOG:
- 2026-07-24  created — WHY: Phase 2 Ghost Trades for evidence-backed IRAC.
# NEXT EDITOR: append dated WHY; keep this line.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np

from telemetry.mind_probe import DecisionRecord, DayMindDump, OP_NAMES

LONG_OPS = {1, 3, 9}
SHORT_OPS = {2, 4, 10}
HOLD = 0


@dataclass
class GhostTrade:
    t: int
    chosen_op: int
    chosen_op_name: str
    alt_op: int
    alt_op_name: str
    pattern: str
    rationale: str
    pattern_alignment: float
    alt_prob: float
    chosen_prob: float
    high_miss: bool = False


@dataclass
class GhostReport:
    brain_name: str
    day_label: str
    n_ghosts: int = 0
    n_high_miss_pull: int = 0
    ghosts: list[GhostTrade] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pattern_side(rec: DecisionRecord):
    if rec.pull_buy:
        return "pull_buy", +1
    if rec.pull_sell:
        return "pull_sell", -1
    if rec.rev_buy:
        return "rev_buy", +1
    if rec.rev_sell:
        return "rev_sell", -1
    if rec.cont_buy and not rec.mask_buy_blocked:
        return "cont_buy", +1
    if rec.cont_sell and not rec.mask_sell_blocked:
        return "cont_sell", -1
    return None, None


def _preferred_op(side: int) -> int:
    return 1 if side > 0 else 2


def build_ghosts(dump: DayMindDump, min_alt_prob: float = 0.05) -> GhostReport:
    """Build Ghost Trades for decisions where a Gravity pattern was present but
    the policy did not act aligned. High-miss = pull present + hold."""
    report = GhostReport(brain_name=dump.brain_name, day_label=dump.day_label)
    for rec in dump.decisions:
        pattern, side = _pattern_side(rec)
        if pattern is None or side is None:
            continue
        alt = _preferred_op(side)
        alt_prob = float(rec.op_probs[alt]) if alt < len(rec.op_probs) else 0.0
        chosen_prob = float(rec.op_probs[rec.chosen_op]) if rec.chosen_op < len(rec.op_probs) else 0.0
        chosen_aligned = (rec.chosen_op in LONG_OPS) if side > 0 else (rec.chosen_op in SHORT_OPS)
        high_miss = (not chosen_aligned) and rec.chosen_op == HOLD and pattern.startswith("pull")
        if high_miss:
            report.n_high_miss_pull += 1
        if chosen_aligned and not high_miss:
            continue
        report.ghosts.append(GhostTrade(
            t=rec.t,
            chosen_op=rec.chosen_op,
            chosen_op_name=rec.chosen_op_name,
            alt_op=alt,
            alt_op_name=OP_NAMES.get(alt, str(alt)),
            pattern=pattern,
            rationale=(
                f"At t={rec.t}, {pattern} was present in obs. "
                f"Policy chose {rec.chosen_op_name} (p={chosen_prob:.3f}). "
                f"Aligned alt={OP_NAMES.get(alt)} (p={alt_prob:.3f})."
            ),
            pattern_alignment=1.0,
            alt_prob=alt_prob,
            chosen_prob=chosen_prob,
            high_miss=high_miss,
        ))
    report.n_ghosts = len(report.ghosts)
    report.summary = _ghost_summary(report, dump)
    return report


def _ghost_summary(report: GhostReport, dump: DayMindDump) -> str:
    lines = [
        f"Ghost Trades for {dump.brain_name} day {dump.day_label}: "
        f"{report.n_ghosts} counterfactuals, {report.n_high_miss_pull} high-miss pull bars."
    ]
    if report.n_high_miss_pull > 5:
        lines.append(
            "IRAC SIGNAL: repeated pull present + hold. "
            "Candidate Perception (low alt_prob) or Policy (alt_prob present but hold wins)."
        )
    high = [g for g in report.ghosts if g.high_miss]
    if high:
        mean_alt = float(np.mean([g.alt_prob for g in high]))
        if mean_alt < 0.08:
            lines.append(
                f"Perception lean: mean alt_prob on high-miss pull = {mean_alt:.3f}."
            )
        else:
            lines.append(
                f"Policy lean: mean alt_prob on high-miss pull = {mean_alt:.3f}."
            )
    return " ".join(lines)
