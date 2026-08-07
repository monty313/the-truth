"""Offline KAG strategy labelers for multi-head physics teaching.

Strategies produce (act, topology, wait) labels only.
They never mutate the 168-dim observation space.
"""

from lineages.adaptive_rl_brain_7_31_26.strategies.cci_dual_level_continuation import (
    STRATEGY_ID as CCI_CONTINUATION_ID,
    collect_continuation_dataset,
    detect_continuation_at,
)
from lineages.adaptive_rl_brain_7_31_26.strategies.rsi_bb_pullback_continuation import (
    STRATEGY_ID as RSI_BB_ID,
    collect_rsi_bb_dataset,
    detect_rsi_bb_at,
)

__all__ = [
    "CCI_CONTINUATION_ID",
    "RSI_BB_ID",
    "collect_continuation_dataset",
    "collect_rsi_bb_dataset",
    "detect_continuation_at",
    "detect_rsi_bb_at",
]
