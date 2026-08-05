"""Pin MARK SETS LAW — LTF first, HTF last two, all four stacks."""
from __future__ import annotations

import pytest

from lineages.adaptive_rl_brain_7_31_26.perception.sets import (
    MARK_SETS_LAW,
    OFFICIAL_SETS,
    assert_mark_sets_law,
    mark_sets_law_table,
)


def test_mark_sets_law_tuple_matches_monty():
    assert len(MARK_SETS_LAW) == 4
    stacks = [(ltf, h0, h1) for _sid, _n, ltf, (h0, h1) in MARK_SETS_LAW]
    assert stacks == [
        ("1m", "15m", "30m"),
        ("5m", "30m", "1h"),
        ("15m", "1h", "4h"),
        ("30m", "4h", "1d"),
    ]


def test_official_sets_built_from_law():
    assert_mark_sets_law()
    assert len(OFFICIAL_SETS) == 4
    for s, law in zip(OFFICIAL_SETS, MARK_SETS_LAW):
        sid, name, ltf, htfs = law
        assert s.set_id == sid
        assert s.name == name
        assert s.entry_tf == ltf
        assert s.confirmation_tfs == htfs
        assert s.tfs == (ltf, htfs[0], htfs[1])


def test_ltf_job_is_entry_htf_is_confirm():
    table = mark_sets_law_table()
    for row in table:
        assert row["ltf_entry"] == row["stack"][0]
        assert row["htf_confirm"] == row["stack"][1:]
        assert row["ltf_job"] == "pullback_continuation_add"
        assert row["htf_job"] == "trend_confirm"


def test_assert_rejects_wrong_stack():
    with pytest.raises(AssertionError, match="MARK SETS LAW broken"):
        assert_mark_sets_law(
            [
                ("1m", "15m", "30m"),
                ("5m", "1h", "4h"),  # wrong — not 5m,30m,1h
                ("15m", "1h", "4h"),
                ("30m", "4h", "1d"),
            ]
        )


def test_mark_doctrine_and_opportunity_import_law():
    """Mark-on-chart path must load sets that pass law (import side-effect)."""
    from lineages.adaptive_rl_brain_7_31_26.perception import mark_doctrine
    from lineages.adaptive_rl_brain_7_31_26.perception import mark_sets_opportunity

    assert_mark_sets_law()
    assert mark_doctrine is not None
    assert mark_sets_opportunity is not None
