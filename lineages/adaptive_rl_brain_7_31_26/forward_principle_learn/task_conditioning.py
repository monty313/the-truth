"""Runtime target%/risk% as task context — train across a grid, no retrain at live.

GOAL.md: same brain solves whatever Monty types. Training must see a
distribution of tasks so attention generalizes inside the grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

# Yardstick pairs used across the lab (soft → hard)
DEFAULT_TASK_GRID: Tuple[Tuple[float, float], ...] = (
    (1.0, 2.0),
    (1.5, 2.0),
    (1.5, 3.0),
    (2.0, 2.5),
    (2.0, 3.0),
    (2.0, 3.5),
    (2.5, 3.5),
    (3.0, 3.5),
)


@dataclass(frozen=True)
class Task:
    target_pct: float
    risk_pct: float

    @property
    def hardness(self) -> str:
        if self.target_pct < 1.75:
            return "soft"
        if self.target_pct < 2.5:
            return "mid"
        return "hard"

    @property
    def pair_key(self) -> str:
        return f"{self.target_pct:g}/{self.risk_pct:g}"


def sample_task_grid(
    n: int = 16,
    *,
    grid: Sequence[Tuple[float, float]] | None = None,
    seed: int = 42,
) -> List[Task]:
    """Sample tasks with replacement from official grid (practice distribution)."""
    g = list(grid or DEFAULT_TASK_GRID)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=max(1, n))
    return [Task(float(g[i][0]), float(g[i][1])) for i in idx]


def task_vector(task: Task) -> np.ndarray:
    """Normalized task embedding for student obs concat / FiLM-style inputs.

    Live path: rebuild this from operator knobs — never bake one forever pair
    into weights.
    """
    t = float(task.target_pct)
    r = float(task.risk_pct)
    # Scale into roughly [0,1] for typical challenge ranges
    t_n = np.clip(t / 4.0, 0.0, 1.5)
    r_n = np.clip(r / 5.0, 0.0, 1.5)
    # Derived: room to thrash (hard targets need cleaner path)
    thrash_budget = np.clip(1.0 - (t_n * 0.7), 0.05, 1.0)
    danger_scale = np.clip(1.0 / max(r_n, 0.05), 0.2, 5.0) / 5.0
    hardness = {"soft": 0.0, "mid": 0.5, "hard": 1.0}[task.hardness]
    return np.asarray([t_n, r_n, thrash_budget, danger_scale, hardness], dtype=np.float32)


def aggression_prior(task: Task) -> dict:
    """Principle prior given task — not a trade answer.

    Hard targets: fewer entries, higher topology quality, bank pressure earlier.
    Soft: single-set scalp more allowed; bank can be early thrash-forgiving.
    """
    if task.hardness == "hard":
        return {
            "max_entries_feel": 4,
            "allow_soft_single_set": False,
            "require_multi_set_for_fire": True,
            "wait_loaded_value": "high",
            "bank_progress_threshold": 0.85,
            "principle_ids": [
                "hard_target_quality_over_thrash",
                "wait_is_skill",
                "finish_line_goal_heat",
            ],
        }
    if task.hardness == "mid":
        return {
            "max_entries_feel": 6,
            "allow_soft_single_set": False,
            "require_multi_set_for_fire": True,
            "wait_loaded_value": "high",
            "bank_progress_threshold": 0.9,
            "principle_ids": ["dual_period_tension", "wait_is_skill", "finish_line_goal_heat"],
        }
    return {
        "max_entries_feel": 8,
        "allow_soft_single_set": True,
        "require_multi_set_for_fire": False,
        "wait_loaded_value": "medium",
        "bank_progress_threshold": 0.95,
        "principle_ids": ["dominant_trends", "finish_line_goal_heat"],
    }


def iter_all_grid(grid: Sequence[Tuple[float, float]] | None = None) -> Iterable[Task]:
    for t, r in grid or DEFAULT_TASK_GRID:
        yield Task(float(t), float(r))
