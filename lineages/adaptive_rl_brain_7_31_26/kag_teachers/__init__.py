"""KAG teachers for Mark meta-RL student (the-truth side).

Additive layer. Does not rewrite perception/sets.py or PROVEN.
Doctrine: references/doctrine/kag_mark_doctrine/
ARMY twin: markos_core.kag_mark
"""

from __future__ import annotations

from lineages.adaptive_rl_brain_7_31_26.kag_teachers.decision_chain import (
    bread_and_butter_obs,
    decision_chain,
)
from lineages.adaptive_rl_brain_7_31_26.kag_teachers.lesson import (
    build_lesson,
    validate_lesson,
)
from lineages.adaptive_rl_brain_7_31_26.kag_teachers.novel_protocol import assign_role
from lineages.adaptive_rl_brain_7_31_26.kag_teachers.student_interface import (
    StudentAuxHeads,
    check_learn_not_copy,
    train_step_one_bar,
)
from lineages.adaptive_rl_brain_7_31_26.kag_teachers.teachers import teach_one_bar
from lineages.adaptive_rl_brain_7_31_26.perception.sets import assert_mark_sets_law

# Teacher 2 bridge (ARMY Reason Teacher) — optional partner for policy=Mark
try:
    from lineages.adaptive_rl_brain_7_31_26.kag_teachers.reason_teacher_bridge import (
        partner_cycle,
        reason_with_army,
    )
except Exception:  # pragma: no cover
    reason_with_army = None  # type: ignore
    partner_cycle = None  # type: ignore

__all__ = [
    "StudentAuxHeads",
    "assert_mark_sets_law",
    "assign_role",
    "bread_and_butter_obs",
    "build_lesson",
    "check_learn_not_copy",
    "decision_chain",
    "partner_cycle",
    "reason_with_army",
    "teach_one_bar",
    "train_step_one_bar",
    "validate_lesson",
]
