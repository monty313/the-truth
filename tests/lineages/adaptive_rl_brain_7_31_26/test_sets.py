"""Phase 1 pins: Official Sets + Sub-Sets."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lineages.adaptive_rl_brain_7_31_26.perception.sets import (
    OFFICIAL_SETS,
    SUB_SETS,
    all_official,
    all_subs,
    confirmation_tfs,
    entry_tf,
    get_official,
    get_sub,
)


def test_official_sets_four_exactly_three_tfs_ltf_first():
    assert len(OFFICIAL_SETS) == 4
    expected = {
        1: ("1m", "15m", "30m"),
        2: ("5m", "30m", "1h"),
        3: ("15m", "1h", "4h"),
        4: ("30m", "4h", "1d"),
    }
    for s in OFFICIAL_SETS:
        assert s.set_id in expected
        assert len(s.tfs) == 3
        assert s.tfs == expected[s.set_id]
        assert s.entry_tf == s.tfs[0]
        assert s.confirmation_tfs == s.tfs[1:]
        assert len(s.confirmation_tfs) == 2


def test_sub_sets_a_to_e_correct_pairs():
    assert len(SUB_SETS) == 5
    expected = {
        "A": ("1m", "5m"),
        "B": ("5m", "15m"),
        "C": ("15m", "30m"),
        "D": ("1h", "4h"),
        "E": ("4h", "1d"),
    }
    for s in SUB_SETS:
        assert s.sub_id in expected
        assert s.tfs == expected[s.sub_id]
        assert s.entry_tf == s.tfs[0]
        assert s.confirmation_tf == s.tfs[1]


def test_official_and_sub_frozen_hashable():
    o1 = get_official(1)
    o1b = get_official(1)
    assert o1 == o1b
    assert hash(o1) == hash(o1b)
    # usable as set/dict keys
    assert {o1, o1b} == {o1}
    assert {get_sub("A"), get_sub("a")} == {get_sub("A")}

    # frozen: mutation must fail
    try:
        o1.set_id = 99  # type: ignore[misc]
        raised = False
    except Exception:
        raised = True
    assert raised


def test_lookup_helpers_and_roles():
    s2 = get_official(2)
    assert entry_tf(s2) == "5m"
    assert confirmation_tfs(s2) == ("30m", "1h")
    sub_c = get_sub("C")
    assert entry_tf(sub_c) == "15m"
    assert confirmation_tfs(sub_c) == ("30m",)
    assert len(all_official()) == 4
    assert len(all_subs()) == 5


def test_unknown_set_raises():
    try:
        get_official(9)
        ok = False
    except KeyError:
        ok = True
    assert ok
    try:
        get_sub("Z")
        ok = False
    except KeyError:
        ok = True
    assert ok


if __name__ == "__main__":
    test_official_sets_four_exactly_three_tfs_ltf_first()
    test_sub_sets_a_to_e_correct_pairs()
    test_official_and_sub_frozen_hashable()
    test_lookup_helpers_and_roles()
    test_unknown_set_raises()
    print("test_sets OK")
