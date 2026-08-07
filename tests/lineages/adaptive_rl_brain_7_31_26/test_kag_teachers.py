"""KAG teachers layer — sets law, chain, wait, novel, learn≠copy (the-truth)."""

from __future__ import annotations

from lineages.adaptive_rl_brain_7_31_26.kag_teachers.decision_chain import (
    bread_and_butter_obs,
    decision_chain,
)
from lineages.adaptive_rl_brain_7_31_26.kag_teachers.lesson import validate_lesson
from lineages.adaptive_rl_brain_7_31_26.kag_teachers.novel_protocol import assign_role
from lineages.adaptive_rl_brain_7_31_26.kag_teachers.student_interface import (
    StudentAuxHeads,
    check_learn_not_copy,
    train_step_one_bar,
)
from lineages.adaptive_rl_brain_7_31_26.kag_teachers.teachers import teach_one_bar
from lineages.adaptive_rl_brain_7_31_26.perception.sets import (
    OFFICIAL_SETS,
    assert_mark_sets_law,
)


def test_assert_mark_sets_law_unchanged():
    assert_mark_sets_law()
    stacks = [s.tfs for s in OFFICIAL_SETS]
    assert stacks == [
        ("1m", "15m", "30m"),
        ("5m", "30m", "1h"),
        ("15m", "1h", "4h"),
        ("30m", "4h", "1d"),
    ]


def test_decision_order_fixed():
    chain = decision_chain(obs=bread_and_butter_obs())
    assert chain["chain_order"] == [
        "tide",
        "regime",
        "breath_launch",
        "act",
        "finish",
    ]


def test_wait_loaded_not_freeze():
    chain = decision_chain(obs=bread_and_butter_obs(set_id=2, side=1))
    s2 = next(s for s in chain["sets"] if s["set_id"] == 2)
    assert s2["act"] == "wait_loaded"
    assert s2["wait_subtype"] == "loaded"


def test_novel_never_defines_tide_alone():
    a = assign_role(
        {"name": "NEW_OSC", "period": 14, "tf": "5m", "novel": True},
        known_roles=[],
    )
    assert a.mask_tide is True
    assert a.known_force_wins is True


def test_bread_and_butter_then_fire():
    load = decision_chain(obs=bread_and_butter_obs(set_id=2, side=1))
    assert next(s for s in load["sets"] if s["set_id"] == 2)["act"] == "wait_loaded"
    release_obs = {
        "sets": {
            2: {
                "force_side": 1,
                "inertia_with": True,
                "velocity_against": False,
                "velocity_with": True,
                "g_fixed": True,
                "efficiency_ok": True,
                "regime": "bull_trend",
            }
        }
    }
    rel = decision_chain(obs=release_obs)
    assert next(s for s in rel["sets"] if s["set_id"] == 2)["act"] == "fire_buy"


def test_learn_not_copy_gate():
    assert check_learn_not_copy(
        act_match=0.95, topology_match=0.2, role_map_match=0.1
    )["pass"] is False
    assert check_learn_not_copy(
        act_match=0.9, topology_match=0.9, role_map_match=0.85
    )["pass"] is True


def test_teach_emits_principle_application():
    out = teach_one_bar(bread_and_butter_obs())
    lesson = out["primary_lesson"]
    assert not validate_lesson(lesson)
    student = StudentAuxHeads(
        act=lesson["act"],
        topology=lesson["topology"],
        role_map={s["name"]: s["role"] for s in lesson["sensors"] if s.get("name")},
        wait_subtype=lesson.get("wait_subtype") or "loaded",
    )
    step = train_step_one_bar(lesson, student)
    assert step["learn_gate"]["pass"] is True


def test_doctrine_pack_present():
    from pathlib import Path

    # tests/lineages/adaptive_rl_brain_7_31_26/this → parents[3] = repo root
    root = Path(__file__).resolve().parents[3]
    pack = root / "references" / "doctrine" / "kag_mark_doctrine"
    assert (pack / "MERGE_AND_PRESERVE.md").is_file()
    assert (pack / "schema.yaml").is_file()
    assert (pack / "seed_triples.jsonl").is_file()
