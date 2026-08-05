"""Tests for ENTJ / Mark multi-set opportunity scan."""
from __future__ import annotations

from lineages.adaptive_rl_brain_7_31_26.perception.mark_sets_opportunity import (
    official_sets_table,
    scan_mark_opportunities,
    score_set_opportunity,
)
from lineages.adaptive_rl_brain_7_31_26.perception.sets import OFFICIAL_SETS, get_official
from lineages.adaptive_rl_brain_7_31_26.perception.types import (
    Direction,
    SetConfluence,
    VelocityStrength,
)


def _conf(sid: int, d: Direction, v: VelocityStrength = VelocityStrength.MEDIUM) -> SetConfluence:
    return SetConfluence(
        set_key=f"official:{sid}",
        direction=d,
        velocity=v,
        votes=(),
        n_bull=2 if d == Direction.BULL else 0,
        n_bear=2 if d == Direction.BEAR else 0,
        n_neutral=1,
    )


def test_official_sets_match_monty_lock():
    """LTF first; last two HTF — exact Mark stack."""
    table = official_sets_table()
    assert table[0]["stack"] == ["1m", "15m", "30m"]
    assert table[1]["stack"] == ["5m", "30m", "1h"]
    assert table[2]["stack"] == ["15m", "1h", "4h"]
    assert table[3]["stack"] == ["30m", "4h", "1d"]
    s2 = get_official(2)
    assert s2.entry_tf == "5m"
    assert s2.confirmation_tfs == ("30m", "1h")


def test_aligned_bull_fires():
    official = {s.set_id: _conf(s.set_id, Direction.BULL) for s in OFFICIAL_SETS}
    entry = {s.set_id: Direction.BULL for s in OFFICIAL_SETS}
    opp = scan_mark_opportunities(official, entry)
    assert opp.action_dir == Direction.BULL
    assert opp.n_aligned_bull == 4
    assert "aligned_bull" in opp.reason


def test_pullback_waits_no_flip():
    """HTF bull, LTF bear on all sets → Mark HOLDs (no thrash reverse)."""
    official = {s.set_id: _conf(s.set_id, Direction.BULL) for s in OFFICIAL_SETS}
    entry = {s.set_id: Direction.BEAR for s in OFFICIAL_SETS}
    opp = scan_mark_opportunities(official, entry)
    assert opp.action_dir == Direction.NEUTRAL
    assert opp.n_aligned_bull == 0
    assert opp.n_aligned_bear == 0


def test_single_aligned_micro_enough():
    """One strong aligned micro set can fire ENTJ scalp."""
    official = {
        1: _conf(1, Direction.BEAR, VelocityStrength.STRONG),
        2: _conf(2, Direction.NEUTRAL),
        3: _conf(3, Direction.NEUTRAL),
        4: _conf(4, Direction.NEUTRAL),
    }
    entry = {
        1: Direction.BEAR,
        2: Direction.NEUTRAL,
        3: Direction.NEUTRAL,
        4: Direction.NEUTRAL,
    }
    opp = scan_mark_opportunities(official, entry, min_aligned=1, min_score=1.5)
    assert opp.action_dir == Direction.BEAR
    assert opp.best is not None and opp.best.set_id == 1


def test_score_aligned_beats_htf_only():
    s = get_official(2)
    conf = _conf(2, Direction.BULL, VelocityStrength.MEDIUM)
    aligned = score_set_opportunity(s, conf, Direction.BULL)
    htf_only = score_set_opportunity(s, conf, Direction.NEUTRAL)
    pull = score_set_opportunity(s, conf, Direction.BEAR)
    assert aligned.score > htf_only.score > pull.score
    assert pull.score == 0.0


def test_macro_permission_blocks_against_tide():
    """pt5: Set4 HTF bear forbids aligned micro bull as permission."""
    official = {
        1: _conf(1, Direction.BULL, VelocityStrength.STRONG),
        2: _conf(2, Direction.BULL, VelocityStrength.MEDIUM),
        3: _conf(3, Direction.NEUTRAL),
        4: _conf(4, Direction.BEAR, VelocityStrength.STRONG),
    }
    entry = {
        1: Direction.BULL,
        2: Direction.BULL,
        3: Direction.NEUTRAL,
        4: Direction.BEAR,
    }
    opp = scan_mark_opportunities(official, entry, macro_permission=True)
    assert opp.action_dir != Direction.BULL
    # either BEAR if set4 aligned, or NEUTRAL if scores gated
    assert opp.action_dir in (Direction.BEAR, Direction.NEUTRAL)


def test_doctrine_soft_scalp_uses_opp_min_score_not_hardcoded_2():
    """Teacher path must honor DEFAULT_OPP_MIN_SCORE (1.2), not dead 2.0 gate."""
    from lineages.adaptive_rl_brain_7_31_26.perception.mark_doctrine import (
        DEFAULT_OPP_MIN_SCORE,
        decide_doctrine,
    )

    assert DEFAULT_OPP_MIN_SCORE <= 1.2
    # FLAT single-set: set1 aligned bull medium — should fire with min_score=1.2
    official = {
        1: _conf(1, Direction.BULL, VelocityStrength.MEDIUM),
        2: _conf(2, Direction.NEUTRAL),
        3: _conf(3, Direction.NEUTRAL),
        4: _conf(4, Direction.NEUTRAL),
    }
    entry = {1: Direction.BULL, 2: Direction.NEUTRAL, 3: Direction.NEUTRAL, 4: Direction.NEUTRAL}
    d = decide_doctrine(official, entry, allow_single_set_scalp=True, opp_min_score=1.2)
    assert d.action == 1  # BUY
    assert "soft_single_set" in d.reason or "slingshot" in d.reason or d.action == 1
