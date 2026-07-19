"""Reward doctrine tests (ADR-0005). 5W+I: see test_shell.py header."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.rewards import RewardEngine

class D:  # minimal DayResult stand-in
    def __init__(self, pnl, goal, breach, trades):
        self.pnl_pct, self.goal_hit, self.breached = pnl, goal, breach
        self.closed_trades = trades
        self.equity_curve = [0.0, pnl]

def T(pnl_pct, adds=0, adverse=0.0, probe=False):
    return {"pnl": pnl_pct, "pnl_pct": pnl_pct, "adds": adds,
            "max_adverse": adverse, "stack_green": pnl_pct > 0,
            "probe": probe, "bars": 5, "why": "policy_close"}

def test_pay_only_on_close():
    re = RewardEngine()
    assert re.on_step([], acted=True, anti_gravity=False) == 0.0

def test_closed_trade_pays():
    re = RewardEngine()
    r = re.on_step([T(0.5)], acted=True, anti_gravity=False)
    assert r > 0

def test_streak_climbs_and_resets():
    re = RewardEngine()
    re.record_win_pct = 99.0        # silence the trophy bonus; isolate the streak
    r1, _ = re.on_day_end(D(3.0, True, False, [T(1.0)]))
    r2, _ = re.on_day_end(D(3.0, True, False, [T(1.0)]))
    assert re.streak_days == 2 and r2 > r1        # forever-climbing bonus
    re.on_day_end(D(1.0, False, False, [T(0.2)]))
    assert re.streak_days == 0

def test_death_penalty():
    re = RewardEngine()
    r, info = re.on_day_end(D(-4.2, False, True, [T(-2.0)]))
    assert r < -5 and info.get("death")

def test_record_only_on_won_days():
    re = RewardEngine()
    _, i1 = re.on_day_end(D(1.0, False, False, [T(2.0)]))
    assert "new_record_win_pct" not in i1          # lost day: no trophy
    _, i2 = re.on_day_end(D(3.0, True, False, [T(2.0)]))
    assert i2.get("new_record_win_pct") == 2.0
