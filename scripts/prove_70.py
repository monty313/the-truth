"""THE 70% VERDICT — is +70% in ONE day possible with max drawdown 4.5%?
Constraint set (Monty, 2026-07-21): any strategy, any lot size, leverage 1:100.
Method (a concrete, replayable HINDSIGHT strategy — the Part One oracle standard):
walk the day on the bot's own 5-minute decision grid; for each window, take the
winning direction ONLY if profitable after round-trip spread; size the trade so its
worst INTRABAR adverse excursion (from the window's real high/low) costs at most
4.5% of current equity (leverage capped at 100:1); compound. Because only winning
windows are taken and every trade's loss potential is capped at 4.5%, the day's
equity NEVER dips more than 4.5% below its running peak (or its start) — the floor
law holds by construction. Every number derives from the committed M1 data.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from core.configs import path as rpath
from training.gpu_data import build_day_tensors

L_MAX, DD = 100.0, 0.045
do, dp, dl, dates, cols = build_day_tensors(rpath("data","XAUUSD_curriculum_2026.csv"),
    cache_path=rpath("artifacts","gpu_cache_XAUUSD_curriculum_2026.npz"), verbose=False)
D = do.shape[0]

def day_bound(i, grid=5):
    n = int(dl[i])
    hi, lo, close, spts = dp[i,:n,0], dp[i,:n,1], dp[i,:n,2], dp[i,:n,3]
    sp = float(spts.mean())*0.01
    eq = 1.0; trades = 0; lev_used = []; t70 = None
    for w in range(1, n//grid):
        a, b = (w-1)*grid, w*grid
        p0, p1 = float(close[a]), float(close[b])
        whi, wlo = float(hi[a:b+1].max()), float(lo[a:b+1][lo[a:b+1]>0].min())
        move = abs(p1 - p0) - 2*sp
        if move <= 0: continue
        adverse = (p0 - wlo + sp) if p1 > p0 else (whi - p0 + sp)   # worst intrabar excursion
        adverse_frac = max(adverse / p0, 1e-6)
        lev = min(L_MAX, DD / adverse_frac)
        if lev <= 0: continue
        gain = lev * move / p0
        if gain <= 0: continue
        eq *= (1.0 + gain); trades += 1; lev_used.append(lev)
        if t70 is None and eq >= 1.70: t70 = b
    return eq, trades, (np.mean(lev_used) if lev_used else 0), t70, n, sp

rows = []
for i in range(D):
    eq, tr, ml, t70, n, sp = day_bound(i)
    rows.append((i, eq, tr, ml, t70, n, sp))
rows.sort(key=lambda r: -r[1])
ok70 = sum(1 for r in rows if r[1] >= 1.70)
print("days where +70%% was POSSIBLE at <=4.5%% DD, lev<=100: %d of %d" % (ok70, D))
print()
print("TOP 5 (evidence grade):")
for i, eq, tr, ml, t70, n, sp in rows[:5]:
    frac = ("+70%% reached after %d of %d minutes (%.0f%% of the day)" % (t70, n, 100*t70/n)) if t70 else "-"
    print("  %s | day multiplier x%.1f (= %+.0f%%) | %d winning windows | avg leverage %.0fx | %s"
          % (dates[i], eq, 100*(eq-1), tr, ml, frac))
print()
print("and the QUIETEST day of the book, for the record:")
qi = min(range(D), key=lambda i: float(dp[i,:int(dl[i]),0].max()-dp[i,:int(dl[i]),1][dp[i,:int(dl[i]),1]>0].min()))
for i, eq, tr, ml, t70, n, sp in rows:
    if i == qi:
        print("  %s | x%.1f (%+.0f%%) | 70%% reached: %s" % (dates[i], eq, 100*(eq-1),
              ("minute %d" % t70) if t70 else "no"))
