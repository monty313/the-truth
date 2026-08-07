"""Principle curriculum — examples of principle *application*, not day oracles.

Each episode teaches transferable physics:
  topology + roles + wait subtype + act family + task (T/R)

Held-out family swap: same topology labels under CCI ↔ RSI ↔ Stoch.
Forward split: practice episodes for train; held-out topologies/tasks for exam.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np

from lineages.adaptive_rl_brain_7_31_26.forward_principle_learn.task_conditioning import (
    Task,
    aggression_prior,
    sample_task_grid,
    task_vector,
)
from lineages.adaptive_rl_brain_7_31_26.kag_teachers.decision_chain import decision_chain
from lineages.adaptive_rl_brain_7_31_26.kag_teachers.novel_protocol import assign_role

# Oscillator families for transfer (name is NOT knowledge; role is)
OSC_FAMILIES = ("CCI", "RSI", "Stochastic", "WPR", "DeMarker")

# pt5 five-law stack (basic knowledge) — always on principle_ids when relevant
PT5_LAWS = (
    "dominant_trends",  # Law 1 HTF permission LTF timing
    "breath_vs_launch",  # Law 2 two markets two rules
    "regime_survival",  # Law 3 state machine
    "capital_preservation",  # Law 4 floor sacred
    "speed_vs_weight",  # Law 5 velocity inside force
)


@dataclass
class PrincipleEpisode:
    """One teachable bar-pack of principles (not a calendar day answer)."""

    episode_id: str
    topology: str
    act: str
    wait_subtype: str | None
    tide: str
    set_id: int
    task: Task
    sensors: List[Dict[str, Any]]
    relations: List[str]
    principle_ids: List[str]
    features: np.ndarray  # fixed-size student input
    split: str  # practice | forward
    family: str  # oscillator family used for velocity
    lesson_type: str = "principle_application"
    not_mode: str = "copy_answer"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_lesson(self) -> Dict[str, Any]:
        return {
            "lesson_type": self.lesson_type,
            "not": self.not_mode,
            "set_id": self.set_id,
            "sensors": self.sensors,
            "relations": self.relations,
            "topology": self.topology,
            "act": self.act,
            "wait_subtype": self.wait_subtype,
            "tide": self.tide,
            "principle_ids": self.principle_ids,
            "goal_link": f"task={self.task.pair_key} hardness={self.task.hardness}",
            "forward_note": "same topology if sensor family swapped",
            "task": {
                "target_pct": self.task.target_pct,
                "risk_pct": self.task.risk_pct,
                "hardness": self.task.hardness,
            },
            "family": self.family,
            "split": self.split,
        }

    def label_vector(self) -> Dict[str, Any]:
        """Multi-head labels for learn≠copy training."""
        return {
            "act": self.act,
            "topology": self.topology,
            "wait_subtype": self.wait_subtype or "none",
            "tide": self.tide,
            "goal_pressure": (
                "bankable"
                if self.act == "bank"
                else ("lagging" if self.task.hardness == "hard" else "on_pace")
            ),
            "role_map": {
                str(s.get("name")): str(s.get("role")) for s in self.sensors if s.get("name")
            },
        }


# Topology generators: market physics knobs → decision_chain
def _obs_for(
    *,
    set_id: int = 2,
    force_side: int,
    inertia_with: bool,
    velocity_against: bool,
    velocity_with: bool,
    g_fixed: bool = True,
    g_flip: bool = False,
    efficiency_ok: bool = True,
    regime: str = "bull_trend",
    family: str = "CCI",
    novel_extra: bool = False,
) -> Dict[str, Any]:
    anchor = {1: "1m", 2: "5m", 3: "15m", 4: "30m"}[set_id]
    support0 = {1: "15m", 2: "30m", 3: "1h", 4: "4h"}[set_id]
    sensors: List[Dict[str, Any]] = [
        {
            "name": "SMA",
            "period": 50,
            "tf": support0,
            "role": "force",
            "novel": False,
            "family": "SMA",
        },
        {
            "name": f"{family}",
            "period": 100,
            "tf": support0,
            "role": "inertia",
            "novel": False,
            "family": family,
        },
        {
            "name": f"{family}",
            "period": 30,
            "tf": anchor,
            "role": "velocity",
            "novel": False,
            "family": family,
        },
    ]
    if novel_extra:
        sensors.append(
            {
                "name": "NEW_OSC",
                "period": 7,
                "tf": anchor,
                "novel": True,
                "family": "UNKNOWN",
            }
        )
        known = [s for s in sensors if not s.get("novel")]
        a = assign_role(sensors[-1], set_context={"anchor": anchor, "support": [support0]}, known_roles=known)
        sensors[-1]["role"] = a.role
        sensors[-1]["why_role"] = a.why_role
        sensors[-1]["mask_tide"] = a.mask_tide
    return {
        "sets": {
            set_id: {
                "force_side": force_side,
                "inertia_with": inertia_with,
                "velocity_against": velocity_against,
                "velocity_with": velocity_with,
                "g_fixed": g_fixed,
                "g_flip": g_flip,
                "efficiency_ok": efficiency_ok,
                "regime": regime,
                "sensors": sensors,
            }
        }
    }


def _feature_from(obs: Dict[str, Any], task: Task, set_id: int = 2) -> np.ndarray:
    """Fixed-size principle features (no calendar day identity)."""
    raw = (obs.get("sets") or {}).get(set_id) or {}
    physics = np.asarray(
        [
            float(raw.get("force_side", 0)),
            1.0 if raw.get("inertia_with") else 0.0,
            1.0 if raw.get("velocity_against") else 0.0,
            1.0 if raw.get("velocity_with") else 0.0,
            1.0 if raw.get("g_fixed") else 0.0,
            1.0 if raw.get("g_flip") else 0.0,
            1.0 if raw.get("efficiency_ok") else 0.0,
            float(set_id) / 4.0,
        ],
        dtype=np.float32,
    )
    tv = task_vector(task)
    # Role histogram soft counts
    roles = {"force": 0.0, "inertia": 0.0, "velocity": 0.0, "novel": 0.0}
    for s in raw.get("sensors") or []:
        r = str(s.get("role", ""))
        if r in roles:
            roles[r] += 1.0
        if s.get("novel"):
            roles["novel"] += 1.0
    role_v = np.asarray([roles[k] for k in ("force", "inertia", "velocity", "novel")], dtype=np.float32)
    return np.concatenate([physics, tv, role_v], axis=0)


SCENARIO_SPECS = [
    # bread-and-butter load
    {
        "name": "bb_load_long",
        "kwargs": dict(
            force_side=1,
            inertia_with=True,
            velocity_against=True,
            velocity_with=False,
            regime="bull_trend",
        ),
        "expect_topo": "slingshot_load",
        "expect_act": "wait_loaded",
        "principles": ["dual_period_tension", "wait_is_skill", "ltf_never_votes_side"],
    },
    # release fire long
    {
        "name": "bb_release_long",
        "kwargs": dict(
            force_side=1,
            inertia_with=True,
            velocity_against=False,
            velocity_with=True,
            regime="bull_trend",
        ),
        "expect_topo": "slingshot_release",
        "expect_act": "fire_buy",
        "principles": ["dual_period_tension", "dominant_trends"],
    },
    # load short
    {
        "name": "bb_load_short",
        "kwargs": dict(
            force_side=-1,
            inertia_with=True,
            velocity_against=True,
            velocity_with=False,
            regime="bear_trend",
        ),
        "expect_topo": "slingshot_load",
        "expect_act": "wait_loaded",
        "principles": ["dual_period_tension", "wait_is_skill"],
    },
    # release fire short
    {
        "name": "bb_release_short",
        "kwargs": dict(
            force_side=-1,
            inertia_with=True,
            velocity_against=False,
            velocity_with=True,
            regime="bear_trend",
        ),
        "expect_topo": "slingshot_release",
        "expect_act": "fire_sell",
        "principles": ["dual_period_tension", "dominant_trends"],
    },
    # chop / efficiency dead
    {
        "name": "chop_mask",
        "kwargs": dict(
            force_side=1,
            inertia_with=True,
            velocity_against=False,
            velocity_with=True,
            efficiency_ok=False,
            regime="chop",
        ),
        "expect_topo": "chop",
        "expect_act": "wait_no_trade",
        "principles": ["efficiency_mask", "regime_survival"],
    },
    # collapse
    {
        "name": "collapse_flip",
        "kwargs": dict(
            force_side=0,
            inertia_with=False,
            velocity_against=False,
            velocity_with=False,
            g_fixed=False,
            g_flip=True,
            regime="reversal_transition",
        ),
        "expect_topo": "collapse",
        "expect_act": "kill",
        "principles": ["dominant_trends", "capital_preservation"],
    },
    # flat force no trade
    {
        "name": "flat_force",
        "kwargs": dict(
            force_side=0,
            inertia_with=False,
            velocity_against=False,
            velocity_with=False,
            regime="undefined",
        ),
        "expect_topo": "chop",
        "expect_act": "wait_no_trade",
        "principles": ["dominant_trends", "wait_is_skill"],
    },
    # novel velocity with known force (load)
    {
        "name": "novel_load_long",
        "kwargs": dict(
            force_side=1,
            inertia_with=True,
            velocity_against=True,
            velocity_with=False,
            regime="bull_trend",
            novel_extra=True,
        ),
        "expect_topo": "slingshot_load",
        "expect_act": "wait_loaded",
        "principles": ["zero_shot_role", "novel_never_defines_tide", "wait_is_skill"],
    },
]


def _build_one(
    spec: Dict[str, Any],
    *,
    family: str,
    task: Task,
    set_id: int,
    split: str,
    eid: str,
) -> PrincipleEpisode:
    kwargs = dict(spec["kwargs"])
    kwargs["family"] = family
    kwargs["set_id"] = set_id
    obs = _obs_for(**kwargs)
    chain = decision_chain(obs=obs)
    sr = next(s for s in chain["sets"] if s["set_id"] == set_id)
    # Prefer chain truth; fall back to expected if set empty
    topology = sr["topology"]
    act = sr["act"]
    # Hard targets: if soft single would fire in thrash — curriculum already uses multi-role
    prior = aggression_prior(task)
    principle_ids = list(
        dict.fromkeys(
            list(PT5_LAWS[:2])  # always tide + breath/launch
            + list(spec["principles"])
            + list(prior["principle_ids"])
            + ["learn_not_copy"]
        )
    )
    if topology == "chop":
        principle_ids.append("regime_survival")
    if task.hardness == "hard":
        principle_ids.append("capital_preservation")
        if act in ("fire_buy", "fire_sell"):
            principle_ids.append("hard_target_quality_over_thrash")

    sensors = list((obs["sets"][set_id].get("sensors") or []))
    relations = []
    raw = obs["sets"][set_id]
    if raw.get("inertia_with"):
        relations.append("inertia_with_tide")
    if raw.get("velocity_against"):
        relations.append("velocity_against")
    if raw.get("velocity_with"):
        relations.append("velocity_with_tide")
    if raw.get("g_fixed"):
        relations.append("G_fixed")
    if not raw.get("efficiency_ok", True):
        relations.append("efficiency_mask")
    feats = _feature_from(obs, task, set_id=set_id)
    return PrincipleEpisode(
        episode_id=eid,
        topology=topology,
        act=act,
        wait_subtype=sr.get("wait_subtype"),
        tide=sr.get("tide") or "flat",
        set_id=set_id,
        task=task,
        sensors=sensors,
        relations=relations,
        principle_ids=principle_ids,
        features=feats,
        split=split,
        family=family,
        meta={"spec": spec["name"], "aggression": prior},
    )


def family_swap_episode(
    base: PrincipleEpisode,
    new_family: str,
) -> PrincipleEpisode:
    """Same physics, different oscillator name → same topology labels required."""
    return _build_one(
        {
            "name": base.meta.get("spec", "swap"),
            "kwargs": {
                "force_side": 1 if base.tide == "long_only" else (-1 if base.tide == "short_only" else 0),
                "inertia_with": "inertia_with_tide" in base.relations,
                "velocity_against": "velocity_against" in base.relations,
                "velocity_with": "velocity_with_tide" in base.relations,
                "g_fixed": "G_fixed" in base.relations,
                "efficiency_ok": "efficiency_mask" not in base.relations,
                "regime": "bull_trend" if base.tide == "long_only" else (
                    "bear_trend" if base.tide == "short_only" else "chop"
                ),
                "novel_extra": any(s.get("novel") for s in base.sensors),
            },
            "principles": base.principle_ids,
        },
        family=new_family,
        task=base.task,
        set_id=base.set_id,
        split=base.split,
        eid=f"{base.episode_id}__swap_{new_family}",
    )


def build_principle_curriculum(
    *,
    seed: int = 42,
    n_tasks_per_scenario: int = 3,
    practice_families: Sequence[str] = ("CCI", "RSI"),
    forward_families: Sequence[str] = ("Stochastic", "WPR"),
    set_ids: Sequence[int] = (2, 3),
) -> List[PrincipleEpisode]:
    """Build practice + forward principle episodes (no calendar day IDs)."""
    rng = np.random.default_rng(seed)
    tasks = sample_task_grid(n_tasks_per_scenario * len(SCENARIO_SPECS), seed=seed)
    episodes: List[PrincipleEpisode] = []
    t_i = 0
    for spec in SCENARIO_SPECS:
        for set_id in set_ids:
            for fam in practice_families:
                task = tasks[t_i % len(tasks)]
                t_i += 1
                eid = f"prac__{spec['name']}__{fam}__s{set_id}__{task.pair_key}"
                ep = _build_one(spec, family=fam, task=task, set_id=set_id, split="practice", eid=eid)
                episodes.append(ep)
            # forward: held-out families + sometimes harder tasks
            for fam in forward_families:
                # bias forward toward harder tasks
                hard_tasks = [Task(3.0, 3.5), Task(2.5, 3.5), Task(2.0, 3.0)]
                task = hard_tasks[int(rng.integers(0, len(hard_tasks)))]
                eid = f"fwd__{spec['name']}__{fam}__s{set_id}__{task.pair_key}"
                ep = _build_one(spec, family=fam, task=task, set_id=set_id, split="forward", eid=eid)
                episodes.append(ep)
    return episodes


def curriculum_stats(episodes: Sequence[PrincipleEpisode]) -> Dict[str, Any]:
    prac = [e for e in episodes if e.split == "practice"]
    fwd = [e for e in episodes if e.split == "forward"]
    return {
        "n_total": len(episodes),
        "n_practice": len(prac),
        "n_forward": len(fwd),
        "topologies_practice": sorted({e.topology for e in prac}),
        "topologies_forward": sorted({e.topology for e in fwd}),
        "families_practice": sorted({e.family for e in prac}),
        "families_forward": sorted({e.family for e in fwd}),
        "feature_dim": int(prac[0].features.shape[0]) if prac else 0,
    }


def episodes_to_jsonable(episodes: Sequence[PrincipleEpisode]) -> List[Dict[str, Any]]:
    out = []
    for e in episodes:
        d = asdict(e)
        d["features"] = e.features.tolist()
        d["task"] = {
            "target_pct": e.task.target_pct,
            "risk_pct": e.task.risk_pct,
            "hardness": e.task.hardness,
        }
        out.append(d)
    return out
