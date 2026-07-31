"""Champion bench — the consistency-winning-rate law (Monty's metric).

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty (his ruling: "whoever has the highest consistency
       winning rate wins" = % of days goal-hit-without-breach; audit R6/
       S12 found this file referenced everywhere but nonexistent).
WHAT:  score_days(): the metric. bench(): evaluate a brain over days
       greedily. compare(): champion vs challenger verdict per the
       ruling, profit as tiebreak.
WHEN:  2026-07-19 (audit round 2).
WHERE: scripts/train_bootcamp.py (second-week proof), the weekly
       champion-vs-challenger ritual (Phase 6+), run cards.
WHY:   The chair goes to the metric, not to the newest brain — protects
       Monty from retrain roulette.
INTERCONNECTED WITH: training/ppo.PPO (play_day greedy), inference/
       loader, experiments/tracker (results into run cards).
----------------------------------------------------------------------

CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  created  — WHY: champion-vs-challenger metric was referenced but missing (audit R6/S12).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
from __future__ import annotations


def score_days(day_infos: list[dict]) -> dict:
    """Monty's metric over a list of day results."""
    n = len(day_infos)
    hits = sum(1 for d in day_infos if d.get("goal_hit"))
    breaches = sum(1 for d in day_infos if d.get("breached"))
    pnl = sum(d.get("pnl_pct", 0.0) for d in day_infos)
    return {"days": n,
            "consistency_winning_rate": hits / n if n else 0.0,
            "breaches": breaches,
            "total_pnl_pct": round(pnl, 3)}


def bench(ppo, n_days: int) -> dict:
    """Greedy evaluation of the CURRENT brain over the env's first n_days."""
    infos = [ppo.play_day(i, greedy=True)[6] for i in range(n_days)]
    s = score_days(infos)
    s["day_details"] = [{k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in d.items() if k != "day_result"}
                        for d in infos]
    return s


def compare(champion_score: dict, challenger_score: dict) -> dict:
    """The chair ruling: highest consistency winning rate wins;
    total profit breaks ties. Returns verdict + reason."""
    c, ch = champion_score, challenger_score
    if ch["consistency_winning_rate"] > c["consistency_winning_rate"]:
        return {"winner": "challenger",
                "reason": f"win rate {ch['consistency_winning_rate']:.3f} > "
                          f"{c['consistency_winning_rate']:.3f}"}
    if (ch["consistency_winning_rate"] == c["consistency_winning_rate"]
            and ch["total_pnl_pct"] > c["total_pnl_pct"]):
        return {"winner": "challenger", "reason": "tie on rate, more profit"}
    return {"winner": "champion", "reason": "challenger did not beat the rate"}
