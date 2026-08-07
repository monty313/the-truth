"""Promotion gates for learn≠copy and forward accuracy.

FAIL if act match high but topology/role_map ~ chance.
PASS held-out family swap.
PASS forward: principle accuracy ↑ and no shell violation flags.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Sequence


@dataclass
class GateReport:
    gate: str
    passed: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def learn_not_copy_gate(
    *,
    act_match: float,
    topology_match: float,
    role_map_match: float,
    wait_match: float = 1.0,
    act_threshold: float = 0.80,
    chance: float = 0.40,
) -> GateReport:
    """COPYING if student only memorizes acts."""
    copying = act_match >= act_threshold and (
        topology_match <= chance or role_map_match <= chance
    )
    # weak wait skill also flags copy-ish freeze
    freeze_copy = act_match >= act_threshold and wait_match <= chance
    passed = (not copying) and (not freeze_copy)
    reason = "OK principles track with act"
    if copying:
        reason = "COPYING: act high, topology/role_map near chance — fail promote"
    elif freeze_copy:
        reason = "COPYING: act high but wait_subtype chance — fail promote"
    return GateReport(
        gate="learn_not_copy",
        passed=passed,
        metrics={
            "act_match": act_match,
            "topology_match": topology_match,
            "role_map_match": role_map_match,
            "wait_match": wait_match,
            "copying": copying,
            "freeze_copy": freeze_copy,
        },
        reason=reason,
    )


def held_out_family_gate(
    pairs: Sequence[Mapping[str, Any]],
    *,
    min_agree: float = 0.85,
) -> GateReport:
    """pairs: [{topology_a, topology_b, act_a, act_b}, ...] from family swaps."""
    if not pairs:
        return GateReport(
            gate="held_out_family_swap",
            passed=False,
            metrics={"n": 0},
            reason="no swap pairs",
        )

    def act_family(a: str) -> str:
        if a in ("fire_buy", "fire_sell"):
            return "fire"
        return a

    topo_ok = sum(1 for p in pairs if p.get("topology_a") == p.get("topology_b"))
    act_ok = sum(
        1
        for p in pairs
        if act_family(str(p.get("act_a"))) == act_family(str(p.get("act_b")))
    )
    n = len(pairs)
    topo_rate = topo_ok / n
    act_rate = act_ok / n
    agree = 0.5 * (topo_rate + act_rate)
    passed = agree >= min_agree
    return GateReport(
        gate="held_out_family_swap",
        passed=passed,
        metrics={
            "n": n,
            "topology_agree": topo_rate,
            "act_family_agree": act_rate,
            "agree": agree,
            "min_agree": min_agree,
        },
        reason="OK transfer across families" if passed else "FAIL family transfer",
    )


def forward_accuracy_gate(
    *,
    practice_acc: float,
    forward_acc: float,
    baseline_forward_acc: float,
    breach_flag: bool = False,
    min_forward: float = 0.70,
    min_delta: float = 0.0,
) -> GateReport:
    """Adopt only if forward principle accuracy improves (or holds above bar)."""
    improved = forward_acc >= baseline_forward_acc + min_delta
    above_bar = forward_acc >= min_forward
    passed = improved and above_bar and (not breach_flag)
    # allow first run: baseline 0
    if baseline_forward_acc <= 0 and above_bar and not breach_flag:
        passed = True
        improved = True
    return GateReport(
        gate="forward_accuracy",
        passed=passed,
        metrics={
            "practice_acc": practice_acc,
            "forward_acc": forward_acc,
            "baseline_forward_acc": baseline_forward_acc,
            "improved": improved,
            "above_bar": above_bar,
            "breach_flag": breach_flag,
            "min_forward": min_forward,
        },
        reason=(
            "OK forward principle accuracy"
            if passed
            else "REJECT: forward not improved or below bar / breach"
        ),
    )


def hard_task_gate(
    *,
    hard_forward_acc: float,
    soft_forward_acc: float,
    min_hard: float = 0.55,
    max_soft_hard_gap: float = 0.45,
) -> GateReport:
    """Hard targets must not collapse while soft looks fine (classic thrash mask)."""
    gap = soft_forward_acc - hard_forward_acc
    passed = hard_forward_acc >= min_hard and gap <= max_soft_hard_gap
    return GateReport(
        gate="hard_task_accuracy",
        passed=passed,
        metrics={
            "hard_forward_acc": hard_forward_acc,
            "soft_forward_acc": soft_forward_acc,
            "gap": gap,
            "min_hard": min_hard,
            "max_soft_hard_gap": max_soft_hard_gap,
        },
        reason="OK hard-task principles" if passed else "FAIL hard-target principle collapse",
    )


def evaluate_all_gates(
    *,
    act_match: float,
    topology_match: float,
    role_map_match: float,
    wait_match: float,
    family_pairs: Sequence[Mapping[str, Any]],
    practice_acc: float,
    forward_acc: float,
    baseline_forward_acc: float,
    hard_forward_acc: float,
    soft_forward_acc: float,
    breach_flag: bool = False,
) -> Dict[str, Any]:
    gates = [
        learn_not_copy_gate(
            act_match=act_match,
            topology_match=topology_match,
            role_map_match=role_map_match,
            wait_match=wait_match,
        ),
        held_out_family_gate(family_pairs),
        forward_accuracy_gate(
            practice_acc=practice_acc,
            forward_acc=forward_acc,
            baseline_forward_acc=baseline_forward_acc,
            breach_flag=breach_flag,
        ),
        hard_task_gate(
            hard_forward_acc=hard_forward_acc,
            soft_forward_acc=soft_forward_acc,
        ),
    ]
    all_pass = all(g.passed for g in gates)
    return {
        "promote": all_pass,
        "decision": "KEEP" if all_pass else "REJECT",
        "gates": [g.to_dict() for g in gates],
        "law": "forward_is_adopt_judge; learn_not_copy; no_shell_touch",
    }
