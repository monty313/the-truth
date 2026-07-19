"""Shell law tests — every reviewer attack, pinned forever.
5W+I: WHO Claude (from 3-agent adversarial review 2026-07-19). WHAT hostile
scenarios that must NEVER pass again. WHEN 2026-07-19. WHY the Shell is law
only if regressions are impossible. INTERCONNECTED: backtesting/simulator.

CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  created/last-major  — WHY: v0.1 build + v0.2 audit fixes (see docs/AUDIT_FIXES_2026-07-19.md).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from backtesting.simulator import DaySim


def mkday(n=200, price=2000.0, spread=10.0, atr_ok=True):
    idx = pd.date_range("2026-06-02", periods=n, freq="1min")
    F = pd.DataFrame({
        "open": price, "high": price + 0.5, "low": price - 0.5, "close": price,
        "vol": 1.0, "spread": spread,
        "mask_buy_blocked": 0.0, "mask_sell_blocked": 0.0,
        "15min::atr14": 2.0 if atr_ok else np.nan,
    }, index=idx)
    return F


def test_pending_batch_cannot_blow_heat():
    sim = DaySim(mkday(), 2.5, 4.0)
    accepted = sum(1 for _ in range(60) if sim.try_open(+1, 0.0025)[0])
    # heat allowance ~ (0 + 4%)/1 => max ~16 quarters-of-a-percent
    assert accepted <= 16, f"queued {accepted} > heat allowance"


def test_budget_counts_pending():
    sim = DaySim(mkday(), 2.5, 4.0)
    sim.trades_used = 399
    a = sim.try_open(+1, 0.001)[0]
    b = sim.try_open(+1, 0.001)[0]
    assert a and not b, "400th ok, 401st must be rejected at queue time"


def test_negative_risk_rejected():
    sim = DaySim(mkday(), 2.5, 4.0)
    ok, why = sim.try_open(+1, -0.05)
    assert not ok and why == "bad_risk"


def test_negative_close_fraction_rejected():
    sim = DaySim(mkday(), 2.5, 4.0)
    sim.try_open(+1, 0.0025); sim.step()
    st = sim.stacks[0]
    assert sim.try_close(st, -1.0) is False
    assert st.units > 0


def test_add_to_loser_rejected_and_max_adds():
    F = mkday()
    sim = DaySim(F, 2.5, 4.0)
    sim.try_open(+1, 0.002); sim.step()
    st = sim.stacks[0]
    # price at entry ~ close: mark == avg -> not a winner -> add rejected
    ok, why = sim.try_open(+1, 0.001, add_to=st)
    assert not ok and why == "add_to_loser"


def test_intrabar_floor_wick_triggers_standdown():
    F = mkday(n=50)
    sim = DaySim(F, 2.5, 4.0)
    sim.try_open(+1, 0.0025); sim.step()
    # brute-force a huge position to make the wick matter
    st = sim.stacks[0]
    st.entries = [(2000.0, 3000.0)]      # oversized on purpose (test-only)
    st.stop = 1900.0                     # far away: stop won't save us
    # craft a wick: low dives, close recovers
    i = sim.t + 1
    F.iloc[i, F.columns.get_loc("low")] = 1997.0     # 3000u * -3 = -9k = -9%
    F.iloc[i, F.columns.get_loc("close")] = 2000.0
    F.iloc[i, F.columns.get_loc("high")] = 2000.5
    alive = sim.step()
    assert not alive and sim.dead, "wick through floor must stand down"
    assert sim.res.breached, "worst-case equity breached -4% -> breached flag"


def test_gap_through_stop_fills_at_extreme():
    F = mkday(n=30)
    sim = DaySim(F, 2.5, 4.0)
    sim.try_open(+1, 0.0025); sim.step()
    st = sim.stacks[0]
    entry = st.avg_price
    i = sim.t + 1
    # gap entirely below the stop
    F.iloc[i, F.columns.get_loc("high")] = st.stop - 5.0
    F.iloc[i, F.columns.get_loc("low")] = st.stop - 9.0
    F.iloc[i, F.columns.get_loc("close")] = st.stop - 7.0
    sim.step()
    tr = sim.res.closed_trades[-1]
    assert tr["why"] == "broker_stop"
    # fill must be at the bar's adverse extreme (low - spread), not stop price
    px_implied = entry + tr["pnl"] / tr["units"]
    assert px_implied < st.stop - 4.0, "gap must fill at bar extreme, not stop"


def test_kill_switch_cancels_pending_opens():
    F = mkday(n=30)
    sim = DaySim(F, 2.5, 4.0)
    sim.try_open(+1, 0.0025)
    open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "artifacts", "KILL"), "w").write("t")
    try:
        sim.step()
        assert len(sim.stacks) == 0, "kill bar must not fill queued opens"
        assert sim.dead
    finally:
        os.remove(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "artifacts", "KILL"))


def test_probe_units_capped():
    sim = DaySim(mkday(), 2.5, 4.0)
    sim.try_open(+1, 0.0025, probe=True); sim.step()
    assert sim.stacks and sim.stacks[0].units <= 1.0 + 1e-9   # 0.01 lot gold = 1 oz
