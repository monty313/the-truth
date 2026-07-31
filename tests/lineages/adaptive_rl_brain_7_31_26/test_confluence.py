"""Phase 1 pins: three-group confluence majority + velocity strength."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lineages.adaptive_rl_brain_7_31_26.perception.confluence import (
    aggregate_confluence,
    confluence_from_confirmation_flags,
    confluence_from_group_directions,
    majority_direction,
    make_group_votes,
    velocity_strength,
    vote_from_confirmation_flags,
)
from lineages.adaptive_rl_brain_7_31_26.perception.types import (
    ConfluenceGroup,
    Direction,
    VelocityStrength,
)

B, R, N = Direction.BULL, Direction.BEAR, Direction.NEUTRAL


def test_majority_2_of_3_bull():
    votes = make_group_votes(B, B, R)
    assert majority_direction(votes) == B
    c = confluence_from_group_directions("official:1", B, B, R)
    assert c.direction == B
    assert c.velocity == VelocityStrength.MEDIUM  # 2 agree with bull
    assert c.n_bull == 2 and c.n_bear == 1


def test_majority_2_of_3_bear():
    votes = make_group_votes(R, B, R)
    assert majority_direction(votes) == R
    c = confluence_from_group_directions("official:2", R, B, R)
    assert c.direction == R
    assert c.velocity == VelocityStrength.MEDIUM


def test_majority_3_of_3_strong():
    c = confluence_from_group_directions("official:4", B, B, B)
    assert c.direction == B
    assert c.velocity == VelocityStrength.STRONG


def test_split_1_1_1_conflict_flat():
    # 1 bull, 1 bear, 1 neutral → no majority
    c = confluence_from_group_directions("sub:A", B, R, N)
    assert c.direction == N
    assert c.velocity == VelocityStrength.NONE


def test_all_neutral_flat():
    c = confluence_from_group_directions("sub:B", N, N, N)
    assert c.direction == N
    assert c.velocity == VelocityStrength.NONE
    assert c.n_neutral == 3


def test_single_non_neutral_is_weak_direction():
    c = confluence_from_group_directions("official:3", B, N, N)
    assert c.direction == B
    assert c.velocity == VelocityStrength.WEAK


def test_velocity_strength_levels_exact_mapping():
    # 3 agree
    assert velocity_strength([B, B, B]) == VelocityStrength.STRONG
    # 2 agree
    assert velocity_strength([B, B, R]) == VelocityStrength.MEDIUM
    assert velocity_strength([B, B, N]) == VelocityStrength.MEDIUM
    # 1 agree
    assert velocity_strength([B, N, N]) == VelocityStrength.WEAK
    assert velocity_strength([B, R, N], direction=B) == VelocityStrength.WEAK
    # 0 or conflict
    assert velocity_strength([N, N, N]) == VelocityStrength.NONE
    assert velocity_strength([B, R, N]) == VelocityStrength.NONE  # dir conflict
    assert velocity_strength([B, B, B], direction=N) == VelocityStrength.NONE


def test_velocity_independent_of_which_groups_voted():
    """Strength counts agreement with majority; group identity does not matter."""
    # Same pattern, different group assignment → same direction + velocity
    a = confluence_from_group_directions("k", B, B, N)  # CCI+RSI bull
    b = confluence_from_group_directions("k", N, B, B)  # RSI+channel bull
    c = confluence_from_group_directions("k", B, N, B)  # CCI+channel bull
    assert a.direction == b.direction == c.direction == B
    assert a.velocity == b.velocity == c.velocity == VelocityStrength.MEDIUM


def test_vote_from_confirmation_flags():
    v = vote_from_confirmation_flags(
        ConfluenceGroup.CCI,
        both_above_on_both_confirmations=True,
        both_below_on_both_confirmations=False,
    )
    assert v.direction == B
    v2 = vote_from_confirmation_flags(
        ConfluenceGroup.RSI,
        both_above_on_both_confirmations=False,
        both_below_on_both_confirmations=True,
    )
    assert v2.direction == R
    v3 = vote_from_confirmation_flags(
        ConfluenceGroup.PRICE_CHANNEL,
        both_above_on_both_confirmations=False,
        both_below_on_both_confirmations=False,
    )
    assert v3.direction == N
    # contradictory → neutral
    v4 = vote_from_confirmation_flags(
        ConfluenceGroup.CCI,
        both_above_on_both_confirmations=True,
        both_below_on_both_confirmations=True,
    )
    assert v4.direction == N


def test_confluence_from_confirmation_flags_end_to_end():
    # All three groups both-above on both confirmations → STRONG bull
    c = confluence_from_confirmation_flags(
        "official:1",
        cci_both_above=True, cci_both_below=False,
        rsi_both_above=True, rsi_both_below=False,
        channel_both_above=True, channel_both_below=False,
    )
    assert c.direction == B
    assert c.velocity == VelocityStrength.STRONG
    # Two above, one mixed → MEDIUM bull
    c2 = confluence_from_confirmation_flags(
        "official:1",
        cci_both_above=True, cci_both_below=False,
        rsi_both_above=True, rsi_both_below=False,
        channel_both_above=False, channel_both_below=False,
    )
    assert c2.direction == B
    assert c2.velocity == VelocityStrength.MEDIUM


def test_aggregate_preserves_votes_tuple():
    votes = make_group_votes(B, R, B)
    c = aggregate_confluence("official:2", votes)
    assert len(c.votes) == 3
    assert c.votes[0].group == ConfluenceGroup.CCI
    assert c.set_key == "official:2"


if __name__ == "__main__":
    test_majority_2_of_3_bull()
    test_majority_2_of_3_bear()
    test_majority_3_of_3_strong()
    test_split_1_1_1_conflict_flat()
    test_all_neutral_flat()
    test_single_non_neutral_is_weak_direction()
    test_velocity_strength_levels_exact_mapping()
    test_velocity_independent_of_which_groups_voted()
    test_vote_from_confirmation_flags()
    test_confluence_from_confirmation_flags_end_to_end()
    test_aggregate_preserves_votes_tuple()
    print("test_confluence OK")
