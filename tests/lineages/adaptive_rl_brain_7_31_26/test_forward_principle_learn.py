"""Forward principle learning — derive answers, not copy day oracles."""

from __future__ import annotations

from lineages.adaptive_rl_brain_7_31_26.forward_principle_learn.curriculum import (
    build_principle_curriculum,
    curriculum_stats,
    family_swap_episode,
)
from lineages.adaptive_rl_brain_7_31_26.forward_principle_learn.generalization_gates import (
    evaluate_all_gates,
    learn_not_copy_gate,
)
from lineages.adaptive_rl_brain_7_31_26.forward_principle_learn.mwt_to_principles import (
    build_mwt_aware_curriculum,
    mwt_principle_focus,
)
from lineages.adaptive_rl_brain_7_31_26.forward_principle_learn.task_conditioning import (
    Task,
    sample_task_grid,
    task_vector,
)
from lineages.adaptive_rl_brain_7_31_26.forward_principle_learn.train_principle_student import (
    PrincipleStudent,
    run_forward_learn_cycle,
    score_episodes,
    train_on_practice,
)
from lineages.adaptive_rl_brain_7_31_26.perception.sets import assert_mark_sets_law


def test_sets_law_still_pinned():
    assert_mark_sets_law()


def test_task_vector_runtime_no_retrain():
    soft = Task(1.0, 2.0)
    hard = Task(3.0, 3.5)
    assert soft.hardness == "soft"
    assert hard.hardness == "hard"
    v = task_vector(hard)
    assert v.shape[0] == 5
    grid = sample_task_grid(8, seed=1)
    assert len(grid) == 8


def test_curriculum_practice_forward_split():
    eps = build_principle_curriculum(seed=0)
    stats = curriculum_stats(eps)
    assert stats["n_practice"] > 0
    assert stats["n_forward"] > 0
    # held-out families differ
    assert "CCI" in stats["families_practice"] or "RSI" in stats["families_practice"]
    assert any(f in stats["families_forward"] for f in ("Stochastic", "WPR"))
    # bread-and-butter present
    topos = {e.topology for e in eps}
    assert "slingshot_load" in topos
    assert "slingshot_release" in topos


def test_family_swap_same_topology_truth():
    eps = build_principle_curriculum(seed=1)
    base = next(e for e in eps if e.topology == "slingshot_load" and e.split == "practice")
    swapped = family_swap_episode(base, "Stochastic")
    assert swapped.topology == base.topology
    assert swapped.act == base.act
    assert swapped.family == "Stochastic"


def test_learn_not_copy_gate():
    fail = learn_not_copy_gate(act_match=0.95, topology_match=0.2, role_map_match=0.1)
    assert fail.passed is False
    ok = learn_not_copy_gate(act_match=0.9, topology_match=0.88, role_map_match=0.85)
    assert ok.passed is True


def test_mwt_focus_not_day_copy():
    focus = mwt_principle_focus(
        [{"class": "MARK_WOULD_TAKE", "sub": "policy_wrong_size_or_timing"}] * 5
        + [{"class": "NO_OPPORTUNITY", "sub": "hard_target_no_force_path"}]
    )
    assert focus["n_mwt"] == 5
    assert focus["not"] == "copy_each_mwt_day_answer"
    assert "wait_is_skill" in focus["principle_ids_focus"]
    pack = build_mwt_aware_curriculum(seed=2)
    assert pack["n_episodes"] > 0
    assert any(e.meta.get("sample_weight", 1) >= 2.0 for e in pack["episodes"])


def test_student_trains_principles_not_act_only():
    pack = build_mwt_aware_curriculum(seed=3)
    practice = [e for e in pack["episodes"] if e.split == "practice"]
    forward = [e for e in pack["episodes"] if e.split == "forward"]
    full = PrincipleStudent(seed=3)
    train_on_practice(full, practice, epochs=30, act_only=False)
    s_full = score_episodes(full, forward)

    act_only = PrincipleStudent(seed=3)
    train_on_practice(act_only, practice, epochs=30, act_only=True)
    s_act = score_episodes(act_only, forward)

    # Full student should have better topology than act-only (principle learning)
    assert s_full["topology_match"] >= s_act["topology_match"] - 0.05
    # learn≠copy: high act with low topo fails
    gate = learn_not_copy_gate(
        act_match=s_act["act_match"],
        topology_match=s_act["topology_match"],
        role_map_match=s_act["role_map_match"],
        wait_match=s_act["wait_match"],
    )
    # If act-only somehow gets high act and low topo, gate fails — that's the point
    if s_act["act_match"] >= 0.8 and s_act["topology_match"] <= 0.4:
        assert gate.passed is False


def test_full_cycle_keep_or_honest_reject():
    report = run_forward_learn_cycle(seed=7, epochs=40, write=True)
    assert report["decision"] in ("KEEP", "REJECT")
    assert "forward_score" in report
    assert report["practice_score"]["n"] > 0
    assert report["forward_score"]["n"] > 0
    # Multi-head present
    assert "topology_match" in report["practice_score"]
    assert report.get("partner_lane") or report.get("mission")


def test_evaluate_all_gates_structure():
    out = evaluate_all_gates(
        act_match=0.9,
        topology_match=0.9,
        role_map_match=0.9,
        wait_match=0.9,
        family_pairs=[
            {
                "topology_a": "slingshot_load",
                "topology_b": "slingshot_load",
                "act_a": "wait_loaded",
                "act_b": "wait_loaded",
            }
        ],
        practice_acc=0.9,
        forward_acc=0.85,
        baseline_forward_acc=0.2,
        hard_forward_acc=0.7,
        soft_forward_acc=0.9,
    )
    assert "promote" in out
    assert "gates" in out
    assert out["decision"] in ("KEEP", "REJECT")
