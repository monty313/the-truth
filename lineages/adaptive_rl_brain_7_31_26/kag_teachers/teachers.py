"""Teacher roster — one constitution, specialist tutors (not vote soul)."""

from __future__ import annotations

from typing import Any, Mapping

from lineages.adaptive_rl_brain_7_31_26.kag_teachers.decision_chain import (
    decision_chain,
    pack_official_sets,
)
from lineages.adaptive_rl_brain_7_31_26.kag_teachers.lesson import build_lesson
from lineages.adaptive_rl_brain_7_31_26.kag_teachers.novel_protocol import assign_role


def teach_one_bar(obs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    packs = pack_official_sets(obs)
    chain = decision_chain(packs)
    lessons = []
    for pack, sr in zip(packs, chain["sets"]):
        sensors = list(pack.get("sensors") or [])
        known = [s for s in sensors if not s.get("novel")]
        enriched = []
        for s in sensors:
            s2 = dict(s)
            if s2.get("novel") or not s2.get("role"):
                a = assign_role(s2, set_context=pack, known_roles=known)
                s2["role"] = a.role
                s2["why_role"] = a.why_role
            enriched.append(s2)
        if not enriched:
            enriched = [
                {"name": "FORCE_HTF", "period": 50, "tf": pack["support"][0], "role": "force"},
                {"name": "VEL_LTF", "period": 14, "tf": pack["anchor"], "role": "velocity"},
            ]
        rel = []
        if pack.get("inertia_with"):
            rel.append("inertia_with_tide")
        if pack.get("velocity_against"):
            rel.append("velocity_against")
        if pack.get("g_fixed"):
            rel.append("G_fixed")
        pids = ["dominant_trends", "ltf_never_votes_side", "learn_not_copy", "wait_is_skill"]
        if any(s.get("novel") for s in enriched):
            pids.extend(["zero_shot_role", "novel_never_defines_tide"])
        lesson = build_lesson(
            set_id=int(sr["set_id"]),
            sensors=enriched,
            relations=rel,
            topology=sr["topology"],
            act=sr["act"],
            principle_ids=pids,
            tide=sr.get("tide"),
            wait_subtype=sr.get("wait_subtype"),
        )
        lessons.append(lesson)
    primary = lessons[0]
    for L in lessons:
        if L["act"] in ("fire_buy", "fire_sell", "wait_loaded"):
            primary = L
            break
    return {"chain": chain, "lessons": lessons, "primary_lesson": primary}
