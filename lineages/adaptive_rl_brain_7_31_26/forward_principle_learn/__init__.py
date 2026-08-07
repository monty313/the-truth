"""Forward principle learning — get right answers WITHOUT cloning day oracles.

Companion to the clone path (other agents may BC Mark day answers).
This stack trains transferable principles so forward clear% climbs under
random target%/risk% without retrain, and without memorizing calendar days.

Law:
  - Practice only for gradient / dial search
  - Forward is sole adopt judge
  - Learn topology + wait + roles; act-only = COPYING (fail gate)
  - Shell / PROVEN / sets never touched
"""

from __future__ import annotations

from lineages.adaptive_rl_brain_7_31_26.forward_principle_learn.curriculum import (
    PrincipleEpisode,
    build_principle_curriculum,
    family_swap_episode,
)
from lineages.adaptive_rl_brain_7_31_26.forward_principle_learn.generalization_gates import (
    GateReport,
    evaluate_all_gates,
    learn_not_copy_gate,
)
from lineages.adaptive_rl_brain_7_31_26.forward_principle_learn.task_conditioning import (
    sample_task_grid,
    task_vector,
)
from lineages.adaptive_rl_brain_7_31_26.forward_principle_learn.train_principle_student import (
    PrincipleStudent,
    run_forward_learn_cycle,
    train_on_practice,
)

__all__ = [
    "GateReport",
    "PrincipleEpisode",
    "PrincipleStudent",
    "build_principle_curriculum",
    "evaluate_all_gates",
    "family_swap_episode",
    "learn_not_copy_gate",
    "run_forward_learn_cycle",
    "sample_task_grid",
    "task_vector",
    "train_on_practice",
]
