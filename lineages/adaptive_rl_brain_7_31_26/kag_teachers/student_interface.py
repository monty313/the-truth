"""Student meta-RL interface — aux heads + learn≠copy gate (the-truth)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class StudentAuxHeads:
    act: str = ""
    topology: str = ""
    tide_per_set: dict = field(default_factory=dict)
    wait_subtype: str = ""
    role_map: dict = field(default_factory=dict)
    goal_pressure: str = ""


def check_learn_not_copy(
    *,
    act_match: float,
    topology_match: float,
    role_map_match: float,
    act_threshold: float = 0.8,
    chance: float = 0.35,
) -> dict[str, Any]:
    copying = act_match >= act_threshold and (
        topology_match <= chance or role_map_match <= chance
    )
    return {
        "gate": "learn_not_copy",
        "pass": not copying,
        "copying": copying,
    }


def train_step_one_bar(
    lesson: Mapping[str, Any],
    student: StudentAuxHeads,
) -> dict[str, Any]:
    act_match = 1.0 if student.act == lesson.get("act") else 0.0
    topo_match = 1.0 if student.topology == lesson.get("topology") else 0.0
    teacher_roles = {
        str(s.get("name")): str(s.get("role")) for s in (lesson.get("sensors") or [])
    }
    if teacher_roles and student.role_map:
        hits = sum(1 for n, r in teacher_roles.items() if student.role_map.get(n) == r)
        role_match = hits / len(teacher_roles)
    else:
        role_match = 0.0 if teacher_roles else 1.0
    return {
        "ok": True,
        "learn_gate": check_learn_not_copy(
            act_match=act_match,
            topology_match=topo_match,
            role_map_match=role_match,
        ),
        "topic": "mark.teacher.lesson.v1",
        "lesson": dict(lesson),
    }
