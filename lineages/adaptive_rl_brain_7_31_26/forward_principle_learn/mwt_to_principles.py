"""Convert MARK_WOULD_TAKE autopsy gaps → principle lessons (not bar copy answers).

Other LLMs may clone day answers. Here we extract *why* the miss class
matters for forward learning: size/timing under task hardness, wait skill,
thrash death — so the student generalizes to unseen days.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from lineages.adaptive_rl_brain_7_31_26.forward_principle_learn.curriculum import (
    PrincipleEpisode,
    build_principle_curriculum,
)
from lineages.adaptive_rl_brain_7_31_26.forward_principle_learn.task_conditioning import Task

_HERE = Path(__file__).resolve().parent
_CKPT = _HERE.parent / "checkpoints"
_AUTOPSY = _CKPT / "mark_consistency" / "AUTOPSY_GAPS__latest.json"
_BEST_50 = _CKPT / "fable_50d_match" / "BEST__latest.json"


def load_autopsy(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or _AUTOPSY
    if not p.is_file():
        return {"days": [], "counts": {}, "source": "missing"}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"days": [], "counts": {}, "source": "bad_json"}


def load_best_50(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or _BEST_50
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def mwt_principle_focus(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize learnable gap principles from autopsy rows."""
    mwt = [
        r
        for r in rows
        if str(r.get("class", r.get("Class", ""))).upper() in ("MARK_WOULD_TAKE", "MWT")
        or str(r.get("gap_class", "")).upper() == "MARK_WOULD_TAKE"
    ]
    no_opp = [
        r
        for r in rows
        if "NO_OPP" in str(r.get("class", r.get("Class", ""))).upper()
        or str(r.get("gap_class", "")).upper() == "NO_OPPORTUNITY"
    ]
    # subclass counts
    subs: Dict[str, int] = {}
    for r in mwt:
        sub = str(r.get("sub", r.get("subclass", "policy_wrong_size_or_timing")))
        subs[sub] = subs.get(sub, 0) + 1

    # dominant principle stack for size/timing MWT
    principles = [
        "wait_is_skill",
        "hard_target_quality_over_thrash",
        "dual_period_tension",
        "finish_line_goal_heat",
        "ltf_never_votes_side",
        "learn_not_copy",
    ]
    if no_opp:
        principles.append("no_opp_hold_not_thrash")

    return {
        "n_mwt": len(mwt),
        "n_no_opp": len(no_opp),
        "subclasses": subs,
        "principle_ids_focus": principles,
        "training_implication": (
            "Oversample slingshot_load→wait_loaded and slingshot_release→fire "
            "under mid/hard tasks; punish thrash act sequences; never force entries on no_opp."
        ),
        "not": "copy_each_mwt_day_answer",
        "yes": "learn_size_timing_topology_under_task",
    }


def enrich_curriculum_for_mwt(
    episodes: Sequence[PrincipleEpisode],
    focus: Mapping[str, Any],
) -> List[PrincipleEpisode]:
    """Reweight/tag practice episodes that teach MWT-relevant principles.

    Does not inject calendar day labels. Tags episodes for sample weight use.
    """
    focus_ids = set(focus.get("principle_ids_focus") or [])
    out: List[PrincipleEpisode] = []
    for ep in episodes:
        overlap = [p for p in ep.principle_ids if p in focus_ids]
        meta = dict(ep.meta)
        meta["mwt_relevant"] = bool(overlap)
        meta["mwt_overlap"] = overlap
        # higher weight for load/release under hard/mid (size-timing craft)
        w = 1.0
        if ep.split == "practice" and ep.topology in (
            "slingshot_load",
            "slingshot_release",
        ):
            w = 2.0
            if ep.task.hardness in ("mid", "hard"):
                w = 3.0
        if ep.act == "wait_loaded":
            w = max(w, 2.5)
        meta["sample_weight"] = w
        # rebuild with meta (dataclass frozen? not frozen)
        ep.meta = meta
        out.append(ep)
    return out


def build_mwt_aware_curriculum(seed: int = 42) -> Dict[str, Any]:
    """Full pipeline: autopsy → focus principles → curriculum tags."""
    autopsy = load_autopsy()
    best = load_best_50()
    # autopsy may be list or dict with days
    rows: List[Mapping[str, Any]] = []
    if isinstance(autopsy, list):
        rows = autopsy  # type: ignore
    elif isinstance(autopsy, dict):
        rows = list(autopsy.get("days") or autopsy.get("rows") or [])
        if not rows and "counts" in autopsy:
            # synthetic rows from counts only
            for _ in range(int(autopsy.get("counts", {}).get("MARK_WOULD_TAKE", 0))):
                rows.append(
                    {
                        "class": "MARK_WOULD_TAKE",
                        "sub": "policy_wrong_size_or_timing",
                    }
                )
            for _ in range(int(autopsy.get("counts", {}).get("NO_OPPORTUNITY", 0))):
                rows.append({"class": "NO_OPPORTUNITY", "sub": "hard_target_no_force_path"})

    # fallback if empty: use BEST gap as signal
    if not rows and best:
        mwt = int(best.get("mwt", 17))
        for _ in range(mwt):
            rows.append({"class": "MARK_WOULD_TAKE", "sub": "policy_wrong_size_or_timing"})

    focus = mwt_principle_focus(rows)
    eps = build_principle_curriculum(seed=seed)
    eps = enrich_curriculum_for_mwt(eps, focus)
    return {
        "focus": focus,
        "best_50": best,
        "n_episodes": len(eps),
        "n_mwt_relevant_practice": sum(
            1 for e in eps if e.split == "practice" and e.meta.get("mwt_relevant")
        ),
        "episodes": eps,
    }
