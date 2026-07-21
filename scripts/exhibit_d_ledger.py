"""EXHIBIT D-1 — the trade-by-trade LEDGER behind the 70% verdict.
Same rules as scripts/prove_70.py (5-min windows on real M1 bars; only spread-profitable
windows; every trade sized so its worst INTRABAR dip costs <= 4.5% of current equity;
leverage capped 100:1; compounding). Prints the actual trades with prices, dips,
leverage, and a running $10,000 account. Deterministic: re-run me, same ledger."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from core.configs import path as rpath
from training.gpu_data import build_day_tensors

L_MAX, DD, START = 100.0, 0.045 * 0.999, 10_000.0   # sized a hair INSIDE the 4.5% law — never touches it
do, dp, dl, dates, cols = build_day_tensors(rpath("data","XAUUSD_curriculum_2026.csv"),
    cache_path=rpath("artifacts","gpu_cache_XAUUSD_curriculum_2026.npz"), verbose=False)

def ledger(date_str, stop_at=1.70, show_rows=True, grid=5):
    i = [k for k,d in enumerate(dates) if date_str in str(d)][0]
    n = int(dl[i]); hi, lo, close, spts = dp[i,:n,0], dp[i,:n,1], dp[i,:n,2], dp[i,:n,3]
    sp = float(spts.mean())*0.01
    eq = 1.0; rows = []; worst_dip = 0.0; t70 = None
    for w in range(1, n//grid):
        a, b = (w-1)*grid, w*grid
        p0, p1 = float(close[a]), float(close[b])
        whi = float(hi[a:b+1].max()); wlo = float(lo[a:b+1][lo[a:b+1]>0].min())
        move = abs(p1-p0) - 2*sp
        if move <= 0: continue
        long = p1 > p0
        adverse = (p0 - wlo + sp) if long else (whi - p0 + sp)
        af = max(adverse/p0, 1e-6)
        lev = min(L_MAX, DD/af)
        gain = lev*move/p0
        if gain <= 0: continue
        dip = lev*af
        worst_dip = max(worst_dip, dip)
        eq *= 1.0+gain
        rows.append((a, b, "LONG " if long else "SHORT", p0, p1, 100*abs(p1-p0)/p0, 100*dip, lev, 100*gain, START*eq))
        if t70 is None and eq >= stop_at:
            t70 = b
            break                                   # stop at the target: trades counted are trades USED
    if show_rows:
        print("LEDGER — %s (spread %.2f, start $%.0f, dip cap 4.5%% of equity):" % (dates[i], sp, START))
        print("  min  side   entry      exit     price-move   worst-dip   leverage   gain     balance")
        for a,b,s,p0,p1,mv,dp_,lv,g,bal in rows:
            print(("  %3d-%-3d %s %9.2f  %9.2f   %+.3f%%      -%.2f%%      %4.0fx    %+5.1f%%   $"
                   % (a,b,s,p0,p1,mv,dp_,lv,g)) + format(bal, ",.0f"))
        print("  +70%% CROSSED at minute %s | worst single dip: -%.2f%% of equity (law: 4.5%%) | trades used: %d"
              % (t70, 100*worst_dip, len(rows)))
    return t70, worst_dip, len(rows), eq

print("="*94)
t70, wd, nt, eq = ledger("2026-01-29")
print()
print("summaries (same rules, run to +70% then stop):")
for d in ("2026-03-23","2026-02-02","2026-01-30","2026-05-22"):
    t70, wd, nt, eq = ledger(d, show_rows=False)
    print("  %s: +70%% at minute %3d | %2d trades used | worst dip -%.2f%% of equity (never past 4.5%%)" % (d, t70, nt, 100*wd))
print()
n70 = 0
for k in range(do.shape[0]):
    t, w_, nt_, e_ = ledger(str(dates[k]), show_rows=False)
    if t is not None: n70 += 1
print("FULL COUNT: days where +70%% was reached under the 4.5%% law: %d of %d" % (n70, do.shape[0]))
