"""Phase 2 Slice 4 + anti-hold: reward dials, credit, EOD did-nothing."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lineages.adaptive_rl_brain_7_31_26.rewards import (
    CORRECT_SIDE_ENTRY_BONUS,
    DEFAULT_DIALS,
    DIAL_BOUNDS,
    DID_NOTHING_EOD_PENALTY,
    DIRECTIONAL_HOLD_PENALTY,
    FLAT_HOLD_TAX,
    FLIP_FLOP_PENALTY,
    INACTIVITY_SETUP_FLOOR,
    MAJORITY_IDLE_PENALTY,
    MAX_OPEN_UNITS,
    MINDLESS_PENALTY,
    REVERSE_COOLDOWN_BARS,
    SECOND_BEST_REGRET_PENALTY,
    STRUCTURE_MATCH_ENTRY_BONUS,
    clip_dials,
    correct_side_entry_bonus,
    credit,
    did_nothing_eod_penalty,
    directional_hold_penalty,
    flat_hold_tax,
    flip_flop_penalty,
    inactivity_penalty,
    majority_agents_idle_penalty,
    make_dials,
    missed_opportunity_penalty,
    pnl_unit,
    second_best_entry_from_logits,
    setup_hold_penalty,
    structure_match_entry_bonus,
)
from lineages.adaptive_rl_brain_7_31_26.perception.types import Direction, TradeTag


def test_dial_bounds_locked():
    assert DIAL_BOUNDS["w_with_vector"] == (0.5, 2.0)
    assert DIAL_BOUNDS["w_qualified_macro"] == (0.5, 2.0)
    assert DIAL_BOUNDS["w_qualified_micro"] == (0.15, 0.7)
    assert DIAL_BOUNDS["w_inactivity"] == (0.0, 1.0)


def test_clip_dials_enforces_bounds():
    c = clip_dials({
        "w_with_vector": 99.0,
        "w_qualified_macro": -5.0,
        "w_qualified_micro": 0.01,
        "w_inactivity": 2.0,
    })
    assert c["w_with_vector"] == 2.0
    assert c["w_qualified_macro"] == 0.5
    assert c["w_qualified_micro"] == 0.15
    assert c["w_inactivity"] == 1.0


def test_pnl_unit_clip():
    assert pnl_unit(50.0, 25.0) == 1.0
    assert pnl_unit(-50.0, 25.0) == -1.0
    assert abs(pnl_unit(5.0, 10.0) - 0.5) < 1e-9
    assert pnl_unit(1.0, 0.0) == 0.0


def test_credit_with_vector_formula():
    dials = make_dials(w_with_vector=1.0)
    # +0.5R → credit 0.5
    assert abs(credit(TradeTag.WITH_VECTOR, 5.0, 10.0, dials) - 0.5) < 1e-9
    dials2 = make_dials(w_with_vector=2.0)
    assert abs(credit(TradeTag.WITH_VECTOR, 5.0, 10.0, dials2) - 1.0) < 1e-9


def test_credit_macro_micro_use_own_weights():
    d = make_dials(w_qualified_macro=2.0, w_qualified_micro=0.2)
    assert abs(credit(TradeTag.QUALIFIED_MACRO, 10.0, 10.0, d) - 2.0) < 1e-9
    assert abs(credit(TradeTag.QUALIFIED_MICRO, 10.0, 10.0, d) - 0.2) < 1e-9


def test_mindless_fixed_penalty_ignores_lucky_pnl():
    d = make_dials(w_with_vector=2.0)
    # Even huge profit → still wall penalty
    assert credit(TradeTag.MINDLESS, 1_000_000.0, 1.0, d) == MINDLESS_PENALTY
    assert credit(TradeTag.MINDLESS, -1_000_000.0, 1.0, d) == MINDLESS_PENALTY


def test_inactivity_penalty():
    d = make_dials(w_inactivity=0.3)
    assert abs(inactivity_penalty(d) - (-0.3)) < 1e-9
    d0 = make_dials(w_inactivity=0.0)
    assert inactivity_penalty(d0) == 0.0


def test_defaults_inside_bounds():
    for k, v in DEFAULT_DIALS.items():
        lo, hi = DIAL_BOUNDS[k]
        assert lo <= v <= hi
    # anti-hold: inactivity default must be non-zero
    assert DEFAULT_DIALS["w_inactivity"] > 0.0


def test_setup_hold_penalty_floor_on_action_tags():
    # dial low → still floor + flat tax for WITH_VECTOR / QUALIFIED_MACRO
    d = make_dials(w_inactivity=0.1)
    p = setup_hold_penalty(TradeTag.WITH_VECTOR, d)
    assert p <= -(INACTIVITY_SETUP_FLOOR + FLAT_HOLD_TAX) + 1e-9
    p2 = setup_hold_penalty(TradeTag.QUALIFIED_MACRO, d)
    assert p2 <= -(INACTIVITY_SETUP_FLOOR + FLAT_HOLD_TAX) + 1e-9
    # micro = dial + tax
    p3 = setup_hold_penalty(TradeTag.QUALIFIED_MICRO, d)
    assert abs(p3 - (-0.1 - FLAT_HOLD_TAX)) < 1e-9
    # mindless / none → flat tax only
    assert abs(setup_hold_penalty(TradeTag.MINDLESS, d) - (-FLAT_HOLD_TAX)) < 1e-9
    assert abs(flat_hold_tax() - (-FLAT_HOLD_TAX)) < 1e-9


def test_correct_side_entry_bonus():
    b = correct_side_entry_bonus(
        TradeTag.WITH_VECTOR, Direction.BULL, Direction.BULL
    )
    assert abs(b - CORRECT_SIDE_ENTRY_BONUS) < 1e-9
    # wrong side
    assert (
        correct_side_entry_bonus(
            TradeTag.WITH_VECTOR, Direction.BULL, Direction.BEAR
        )
        == 0.0
    )
    # mindless never
    assert (
        correct_side_entry_bonus(TradeTag.MINDLESS, Direction.BULL, Direction.BULL)
        == 0.0
    )


def test_did_nothing_eod_penalty():
    assert did_nothing_eod_penalty(0.0, 0) == DID_NOTHING_EOD_PENALTY
    assert did_nothing_eod_penalty(0.0, 1) == 0.0  # entered → no did-nothing
    assert did_nothing_eod_penalty(1.5, 0) == 0.0  # non-zero pnl
    assert DID_NOTHING_EOD_PENALTY <= -15.0
    assert DID_NOTHING_EOD_PENALTY >= -25.0


def test_majority_agents_idle_penalty():
    assert majority_agents_idle_penalty(
        has_majority=True, action_is_hold=True, is_flat=True
    ) == MAJORITY_IDLE_PENALTY
    assert MAJORITY_IDLE_PENALTY < 0.0
    assert majority_agents_idle_penalty(
        has_majority=True, action_is_hold=False, is_flat=True
    ) == 0.0


def test_thrash_control_constants():
    assert MAX_OPEN_UNITS >= 2
    assert MAX_OPEN_UNITS <= 5
    assert REVERSE_COOLDOWN_BARS >= 20
    assert flip_flop_penalty() == -abs(FLIP_FLOP_PENALTY)
    assert FLIP_FLOP_PENALTY < 0.0


def test_directional_hold_penalty_stronger_than_inactivity():
    # Flat + structure BUY + HOLD → strong penalty
    p = directional_hold_penalty(is_flat=True, action=0, structure_rec=1)
    assert p == -abs(DIRECTIONAL_HOLD_PENALTY)
    assert p < setup_hold_penalty(TradeTag.WITH_VECTOR, make_dials())
    # Matching side → no hold penalty
    assert directional_hold_penalty(is_flat=True, action=1, structure_rec=1) == 0.0
    # Structure HOLD → no directional hold penalty
    assert directional_hold_penalty(is_flat=True, action=0, structure_rec=0) == 0.0
    # In trade (not flat) → 0
    assert directional_hold_penalty(is_flat=False, action=0, structure_rec=2) == 0.0


def test_structure_match_entry_bonus():
    assert (
        structure_match_entry_bonus(is_flat=True, action=1, structure_rec=1)
        == abs(STRUCTURE_MATCH_ENTRY_BONUS)
    )
    assert structure_match_entry_bonus(is_flat=True, action=2, structure_rec=1) == 0.0
    assert structure_match_entry_bonus(is_flat=True, action=0, structure_rec=1) == 0.0
    assert structure_match_entry_bonus(is_flat=False, action=1, structure_rec=1) == 0.0


def test_second_best_entry_from_logits():
    # HOLD best, SELL second
    assert second_best_entry_from_logits([1.0, -0.5, 0.2]) == 2
    # HOLD best, BUY second
    assert second_best_entry_from_logits([1.0, 0.5, -0.2]) == 1
    # BUY best, HOLD second → 2nd not entry → None
    assert second_best_entry_from_logits([0.5, 1.0, -1.0]) is None


def test_missed_opportunity_penalty():
    assert missed_opportunity_penalty(cf_pnl_after_fees=1.5) == -abs(
        SECOND_BEST_REGRET_PENALTY
    )
    assert missed_opportunity_penalty(cf_pnl_after_fees=0.0) == 0.0
    assert missed_opportunity_penalty(cf_pnl_after_fees=-0.2) == 0.0
    assert abs(SECOND_BEST_REGRET_PENALTY) < abs(MINDLESS_PENALTY)


if __name__ == "__main__":
    test_dial_bounds_locked()
    test_clip_dials_enforces_bounds()
    test_pnl_unit_clip()
    test_credit_with_vector_formula()
    test_credit_macro_micro_use_own_weights()
    test_mindless_fixed_penalty_ignores_lucky_pnl()
    test_inactivity_penalty()
    test_defaults_inside_bounds()
    test_setup_hold_penalty_floor_on_action_tags()
    test_correct_side_entry_bonus()
    test_did_nothing_eod_penalty()
    test_majority_agents_idle_penalty()
    test_thrash_control_constants()
    test_directional_hold_penalty_stronger_than_inactivity()
    test_structure_match_entry_bonus()
    test_second_best_entry_from_logits()
    test_missed_opportunity_penalty()
    print("test_rewards OK")