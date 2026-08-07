"""principle_application lessons — reject act-only."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


class LessonRejected(ValueError):
    pass


def validate_lesson(lesson: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if lesson.get("lesson_type") != "principle_application":
        errors.append("lesson_type must be principle_application")
    if lesson.get("not") != "copy_answer":
        errors.append("must declare not=copy_answer")
    if not lesson.get("topology"):
        errors.append("topology required")
    if not lesson.get("principle_ids"):
        errors.append("principle_ids required")
    sensors = lesson.get("sensors") or []
    if not sensors:
        errors.append("sensors with roles required")
    if not lesson.get("act"):
        errors.append("act required")
    return errors


def build_lesson(
    *,
    set_id: int,
    sensors: Sequence[Mapping[str, Any]],
    relations: Sequence[str],
    topology: str,
    act: str,
    principle_ids: Sequence[str],
    goal_link: str = "",
    forward_note: str = "same topology if sensor family swapped",
    tide: str | None = None,
    wait_subtype: str | None = None,
) -> dict[str, Any]:
    lesson = {
        "lesson_type": "principle_application",
        "not": "copy_answer",
        "set_id": int(set_id),
        "sensors": [dict(s) for s in sensors],
        "relations": list(relations),
        "topology": topology,
        "act": act,
        "principle_ids": list(principle_ids),
        "goal_link": goal_link or "daily target reachable",
        "forward_note": forward_note,
    }
    if tide is not None:
        lesson["tide"] = tide
    if wait_subtype is not None:
        lesson["wait_subtype"] = wait_subtype
    errs = validate_lesson(lesson)
    if errs:
        raise LessonRejected("; ".join(errs))
    return lesson
