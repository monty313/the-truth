"""500-slot signal suggestion bus (observation channels).

Empty slots emit 0. Filled slots emit -1 (sell), 0 (flat), or +1 (buy).
RL may ignore all of them; they are suggestions in obs, not orders.
"""
from .encode import N_SLOTS, append_signal_obs, signal_column_names

__all__ = ["N_SLOTS", "append_signal_obs", "signal_column_names"]
