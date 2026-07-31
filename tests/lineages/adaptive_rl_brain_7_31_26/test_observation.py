"""Phase 2 Slice 3: Channel 1 observation block."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lineages.adaptive_rl_brain_7_31_26.perception.observation import (
    CHANNEL1_DIM,
    build_channel1_obs,
    channel1_layout,
    confluence_score,
    empty_confluence,
    velocity_to_float,
)
from lineages.adaptive_rl_brain_7_31_26.perception.types import (
    Direction,
    SetConfluence,
    StructureFlags,
    VelocityStrength,
)


def _c(key, d, v, nb=2, nr=0, nn=1):
    return SetConfluence(
        set_key=key, direction=d, velocity=v, votes=(),
        n_bull=nb, n_bear=nr, n_neutral=nn,
    )


def test_channel1_dim_is_32():
    assert CHANNEL1_DIM == 32
    obs = build_channel1_obs()
    assert obs.shape == (32,)
    assert obs.dtype == np.float32


def test_zeros_when_empty():
    obs = build_channel1_obs()
    assert np.allclose(obs, 0.0)


def test_official_and_sub_packing():
    official = {
        1: _c("official:1", Direction.BULL, VelocityStrength.STRONG, 3, 0, 0),
        4: _c("official:4", Direction.BEAR, VelocityStrength.MEDIUM, 0, 2, 1),
    }
    subs = {
        "A": _c("sub:A", Direction.BULL, VelocityStrength.WEAK, 1, 0, 2),
    }
    struct = StructureFlags(pullback=True, scale_conflict=False)
    obs = build_channel1_obs(
        official, subs, struct,
        progress_to_goal=0.5, danger=0.25, session_phase=0.1,
    )
    # Official set 1 at indices 0..2
    assert obs[0] == 1.0  # bull
    assert abs(obs[1] - 1.0) < 1e-6  # strong
    assert abs(obs[2] - 1.0) < 1e-6  # 3/3 bull score
    # Official set 4 at indices 9..11
    assert obs[9] == -1.0
    assert abs(obs[10] - 2.0 / 3.0) < 1e-6
    # Sub A at 12..14
    assert obs[12] == 1.0
    assert abs(obs[13] - 1.0 / 3.0) < 1e-6
    # flags + placeholders
    assert obs[27] == 1.0
    assert obs[28] == 0.0
    assert abs(obs[29] - 0.5) < 1e-6
    assert abs(obs[30] - 0.25) < 1e-6
    assert abs(obs[31] - 0.1) < 1e-6


def test_velocity_mapping():
    assert velocity_to_float(VelocityStrength.NONE) == 0.0
    assert velocity_to_float(VelocityStrength.STRONG) == 1.0


def test_confluence_score_signed():
    c = empty_confluence("x")
    assert confluence_score(c) == 0.0
    bull = _c("b", Direction.BULL, VelocityStrength.STRONG, 3, 0, 0)
    assert confluence_score(bull) == 1.0


def test_layout_docs():
    lay = channel1_layout()
    assert lay["dim"] == 32
    assert "PROVEN" in str(lay["note"])


if __name__ == "__main__":
    test_channel1_dim_is_32()
    test_zeros_when_empty()
    test_official_and_sub_packing()
    test_velocity_mapping()
    test_confluence_score_signed()
    test_layout_docs()
    print("test_observation OK")
