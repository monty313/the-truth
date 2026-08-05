#!/usr/bin/env python3
"""Audit Student Strategy Tests CSV + honest algorithmic proxies on our data.

Sheet is marketing self-report (discretionary BarReplay). We:
1) Parse and stress-test the published numbers (bias, consistency).
2) Where a concept is codable, score direction hit-rate on our M1 CSVs.
3) Verdict for signals agent: add / skip / already covered.

Not financial advice. Measurement only.
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from data_io.loader import read_mt5_m1, resample as mt5_resample  # noqa: E402

CSV_PATH = ROOT / "Student Strategy Tests.xlsx - Student Results.csv"
OUT_DIR = ROOT / "outputs" / "artifacts" / "student_strategy_audit"
# Last N M1 rows per symbol (speed). Full files are ~2M bars.
MAX_ROWS = 180_000
DATA_CANDIDATES = [
    ROOT / "data" / "raw" / "EURUSD_M1_curriculum.csv",
    ROOT / "data" / "raw" / "US30_M1_curriculum.csv",
    ROOT / "data" / "raw" / "XAUUSD_curriculum_2026.csv",
    ROOT / "data" / "raw" / "GBPUSD_M1_curriculum.csv",
]
HORIZONS = (5, 10, 20)
# pandas resample rules used by proxies (match loader TF_RULE loosely)
TF_MAP = {"5min": "5min", "15min": "15min", "30min": "30min"}


def parse_date(s: str) -> datetime | None:
    if not s:
        return None
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            pass
    return None


def load_sheet() -> list[dict]:
    rows: list[dict] = []
    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        for i, row in enumerate(r):
            if i < 3:
                continue
            if not row or not row[0] or row[0] == "Student Name":
                continue
            if len(row) < 6:
                continue
            name, added, start, strat, wr, ret = row[0], row[1], row[2], row[3], row[4], row[5]
            if not strat:
                continue
            try:
                wrn = float(str(wr).replace("%", "").strip())
            except ValueError:
                wrn = None
            try:
                retn = float(str(ret).replace("%", "").strip())
            except ValueError:
                retn = None
            sd, ad = parse_date(start), parse_date(added)
            days = (ad - sd).days if sd and ad else None
            base = name.split("(")[0].strip()
            rows.append(
                {
                    "student": name,
                    "base": base,
                    "strat": strat.strip(),
                    "wr": wrn,
                    "ret": retn,
                    "days": days,
                    "start": start,
                    "added": added,
                }
            )
    return rows


def sheet_audit(rows: list[dict]) -> dict:
    by: dict[str, list] = defaultdict(list)
    for x in rows:
        by[x["strat"]].append(x)

    strat_stats = {}
    for s, xs in by.items():
        wrs = [x["wr"] for x in xs if x["wr"] is not None]
        rets = [x["ret"] for x in xs if x["ret"] is not None]
        if not wrs:
            continue
        strat_stats[s] = {
            "n": len(xs),
            "wr_mean": round(st.mean(wrs), 2),
            "wr_med": round(st.median(wrs), 2),
            "wr_std": round(st.pstdev(wrs), 2) if len(wrs) > 1 else 0.0,
            "wr_min": round(min(wrs), 2),
            "wr_max": round(max(wrs), 2),
            "frac_wr_ge_70": round(sum(1 for w in wrs if w >= 70) / len(wrs), 3),
            "ret_mean": round(st.mean(rets), 2) if rets else None,
            "ret_med": round(st.median(rets), 2) if rets else None,
            "ret_min": round(min(rets), 2) if rets else None,
            "ret_max": round(max(rets), 2) if rets else None,
            "all_ret_positive": all(r > 0 for r in rets) if rets else None,
        }

    dups = defaultdict(list)
    for x in rows:
        dups[(x["base"].lower(), x["strat"].lower())].append(x)
    n_dup_groups = sum(1 for v in dups.values() if len(v) > 1)

    rets_all = [x["ret"] for x in rows if x["ret"] is not None]
    return {
        "n_rows": len(rows),
        "n_strategies": len(strat_stats),
        "n_duplicate_base_strat_groups": n_dup_groups,
        "all_published_returns_positive": all(r > 0 for r in rets_all) if rets_all else None,
        "min_ret": min(rets_all) if rets_all else None,
        "max_ret": max(rets_all) if rets_all else None,
        "has_trade_count": False,
        "has_max_dd": False,
        "has_sample_size": False,
        "has_R_multiple": False,
        "methodology": "TradingView BarReplay discretionary; self-reported; screenshots claimed not attached here",
        "strategies": strat_stats,
        "red_flags": [
            "100% of published returns are positive (survivor / showcase selection bias)",
            "No trade count, no max DD, no expectancy, no R-multiples, no out-of-sample split",
            "Win rate alone is not edge (low WR + high return rows prove payoff asymmetry, not skill)",
            "Duplicate student+strategy rows inflate n",
            "Strategies are discretionary academy methods — not closed-form rules we can paste into encode.py",
            "Marketing header claims 'average win rate over 70%' — only true after selection into this list",
        ],
    }


def load_m1(path: Path, max_rows: int = MAX_ROWS) -> pd.DataFrame | None:
    if not path.exists():
        return None
    # MT5 exports are huge; read_mt5_m1 nrows = first N rows (older history).
    # Prefer tail for "recent" regime: read full path in chunks if needed.
    try:
        # Fast path: read last max_rows via skip for large files
        # Count lines cheaply
        with path.open("rb") as f:
            nlines = sum(1 for _ in f)
        data_rows = max(0, nlines - 1)
        if data_rows <= max_rows:
            df = read_mt5_m1(str(path))
        else:
            # skip early rows (keep header)
            skip = data_rows - max_rows
            raw = pd.read_csv(path, sep=None, engine="python", skiprows=range(1, skip + 1))
            raw.columns = [c.strip().strip("<>").upper() for c in raw.columns]
            ts = pd.to_datetime(
                raw["DATE"].astype(str) + " " + raw["TIME"].astype(str),
                format="mixed",
                errors="coerce",
            )
            df = pd.DataFrame(
                {
                    "open": pd.to_numeric(raw["OPEN"], errors="coerce").to_numpy(),
                    "high": pd.to_numeric(raw["HIGH"], errors="coerce").to_numpy(),
                    "low": pd.to_numeric(raw["LOW"], errors="coerce").to_numpy(),
                    "close": pd.to_numeric(raw["CLOSE"], errors="coerce").to_numpy(),
                    "vol": pd.to_numeric(raw.get("TICKVOL", 1.0), errors="coerce").to_numpy(),
                },
                index=ts,
            ).dropna(subset=["open", "high", "low", "close"])
            df = df[~df.index.isna()].sort_index()
            df = df[~df.index.duplicated(keep="last")]
        if df is None or len(df) < 5000:
            return None
        return df
    except Exception as e:
        print(f"  load fail {path.name}: {e}")
        return None


def hit_rate(side: np.ndarray, close: np.ndarray, h: int) -> dict:
    """side in {-1,0,+1}; correct if sign(close[t+h]-close[t]) matches side when side!=0."""
    n = len(close)
    if n <= h:
        return {"n_fires": 0, "hit": None}
    s = side[:-h]
    ret = close[h:] - close[:-h]
    mask = s != 0
    nf = int(mask.sum())
    if nf == 0:
        return {"n_fires": 0, "hit": None}
    correct = ((s > 0) & (ret > 0)) | ((s < 0) & (ret < 0))
    # ties (ret==0) count as miss
    hit = float(correct[mask].mean())
    return {"n_fires": nf, "hit": round(hit, 4)}


def resample_ohlc(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample M1 -> LTF. Prefer project loader when TF known."""
    key = {"5min": "5min", "15min": "15min", "30min": "30min"}.get(rule)
    if key is not None:
        try:
            return mt5_resample(m1, key)
        except Exception:
            pass
    return (
        m1.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def align_sig(ltf_index: pd.DatetimeIndex, m1_index: pd.DatetimeIndex, sig: pd.Series) -> np.ndarray:
    s = sig.reindex(ltf_index).fillna(0.0)
    # forward-fill last known signal onto M1 via reindex+ffill on union then take m1
    joined = s.reindex(m1_index.union(ltf_index)).sort_index().ffill().reindex(m1_index).fillna(0.0)
    return joined.to_numpy(dtype=np.float64)


def proxy_bollinger(m1: pd.DataFrame, tf: str = "5min", n: int = 20, k: float = 2.0) -> np.ndarray:
    """Classic BB mean-reversion: long touch lower band, short touch upper. Pulse on LTF, sticky ffill."""
    ltf = resample_ohlc(m1, tf)
    mid = ltf["close"].rolling(n).mean()
    std = ltf["close"].rolling(n).std()
    up, lo = mid + k * std, mid - k * std
    long_e = (ltf["low"] <= lo) & (ltf["close"] > lo)
    short_e = (ltf["high"] >= up) & (ltf["close"] < up)
    sig = pd.Series(0.0, index=ltf.index)
    sig[long_e] = 1.0
    sig[short_e] = -1.0
    both = long_e & short_e
    sig[both] = 0.0
    # sticky
    sticky = sig.replace(0.0, np.nan).ffill().fillna(0.0)
    return align_sig(ltf.index, m1.index, sticky)


def proxy_fib_pullback(m1: pd.DataFrame, tf: str = "15min", look: int = 20) -> np.ndarray:
    """Fib-ish: swing high/low lookback; long near 61.8% retrace of up-swing; short opposite."""
    ltf = resample_ohlc(m1, tf)
    hh = ltf["high"].rolling(look).max()
    ll = ltf["low"].rolling(look).min()
    rng = (hh - ll).replace(0, np.nan)
    # position of close in range (0=low, 1=high)
    pos = (ltf["close"] - ll) / rng
    # trend proxy: close vs mid of range on longer window
    mid = (hh + ll) / 2
    uptrend = ltf["close"] > mid
    # buy: uptrend and deep pullback into 0.5-0.7 zone (near 61.8 from high = pos ~0.382 from low? 
    # for up-swing: 61.8% retrace means price fell 61.8% of swing -> pos = 1-0.618 = 0.382
    long_e = uptrend & (pos >= 0.30) & (pos <= 0.45) & (ltf["close"] > ltf["open"])
    short_e = (~uptrend) & (pos >= 0.55) & (pos <= 0.70) & (ltf["close"] < ltf["open"])
    sig = pd.Series(0.0, index=ltf.index)
    sig[long_e] = 1.0
    sig[short_e] = -1.0
    sticky = sig.replace(0.0, np.nan).ffill().fillna(0.0)
    return align_sig(ltf.index, m1.index, sticky)


def proxy_supply_demand(m1: pd.DataFrame, tf: str = "15min", base: int = 3, impulse: float = 1.5) -> np.ndarray:
    """Weak S/D proxy: base candle then impulse away; retest of base zone fires with trend of impulse."""
    ltf = resample_ohlc(m1, tf)
    body = (ltf["close"] - ltf["open"]).abs()
    rng = (ltf["high"] - ltf["low"]).replace(0, np.nan)
    atr = rng.rolling(14).mean()
    # base: small range bars
    is_base = rng < (0.6 * atr)
    # impulse bar after base
    bull_imp = (ltf["close"] - ltf["open"]) > (impulse * atr)
    bear_imp = (ltf["open"] - ltf["close"]) > (impulse * atr)
    # mark zones: after base streak then impulse
    base_run = is_base.rolling(base).sum() >= base
    demand_event = base_run.shift(1).fillna(False) & bull_imp  # demand under
    supply_event = base_run.shift(1).fillna(False) & bear_imp
    # zone levels from the base window (simplified: low/high of prior bar)
    demand_lo = ltf["low"].shift(1).where(demand_event)
    demand_hi = ltf["high"].shift(1).where(demand_event)
    supply_lo = ltf["low"].shift(1).where(supply_event)
    supply_hi = ltf["high"].shift(1).where(supply_event)
    d_lo = demand_lo.ffill()
    d_hi = demand_hi.ffill()
    s_lo = supply_lo.ffill()
    s_hi = supply_hi.ffill()
    # retest: price returns into zone
    in_demand = (ltf["low"] <= d_hi) & (ltf["high"] >= d_lo)
    in_supply = (ltf["high"] >= s_lo) & (ltf["low"] <= s_hi)
    long_e = in_demand & (ltf["close"] > ltf["open"]) & (ltf["close"] > d_hi)
    short_e = in_supply & (ltf["close"] < ltf["open"]) & (ltf["close"] < s_lo)
    sig = pd.Series(0.0, index=ltf.index)
    sig[long_e.fillna(False)] = 1.0
    sig[short_e.fillna(False)] = -1.0
    sticky = sig.replace(0.0, np.nan).ffill().fillna(0.0)
    return align_sig(ltf.index, m1.index, sticky)


def proxy_orderblock(m1: pd.DataFrame, tf: str = "15min") -> np.ndarray:
    """Weak OB proxy: last opposite candle before strong impulse; retest."""
    ltf = resample_ohlc(m1, tf)
    atr = (ltf["high"] - ltf["low"]).rolling(14).mean()
    bull_imp = (ltf["close"] - ltf["open"]) > (1.8 * atr)
    bear_imp = (ltf["open"] - ltf["close"]) > (1.8 * atr)
    # last bearish before bull impulse = bullish OB
    prev_bear = ltf["close"].shift(1) < ltf["open"].shift(1)
    prev_bull = ltf["close"].shift(1) > ltf["open"].shift(1)
    bull_ob = bull_imp & prev_bear
    bear_ob = bear_imp & prev_bull
    ob_lo = ltf["low"].shift(1).where(bull_ob).ffill()
    ob_hi = ltf["high"].shift(1).where(bull_ob).ffill()
    sob_lo = ltf["low"].shift(1).where(bear_ob).ffill()
    sob_hi = ltf["high"].shift(1).where(bear_ob).ffill()
    long_e = (ltf["low"] <= ob_hi) & (ltf["close"] > ob_hi) & (ltf["close"] > ltf["open"])
    short_e = (ltf["high"] >= sob_lo) & (ltf["close"] < sob_lo) & (ltf["close"] < ltf["open"])
    sig = pd.Series(0.0, index=ltf.index)
    sig[long_e.fillna(False)] = 1.0
    sig[short_e.fillna(False)] = -1.0
    sticky = sig.replace(0.0, np.nan).ffill().fillna(0.0)
    return align_sig(ltf.index, m1.index, sticky)


def proxy_mw_pattern(m1: pd.DataFrame, tf: str = "15min", look: int = 30) -> np.ndarray:
    """Very crude M/W (double top/bottom) proxy on LTF pivots."""
    ltf = resample_ohlc(m1, tf)
    c = ltf["close"]
    # simple pivots: local max/min over 3 bars
    hi = ltf["high"]
    lo = ltf["low"]
    piv_hi = (hi.shift(1) > hi.shift(2)) & (hi.shift(1) > hi) & (hi.shift(1) > hi.shift(3))
    piv_lo = (lo.shift(1) < lo.shift(2)) & (lo.shift(1) < lo) & (lo.shift(1) < lo.shift(3))
    # double bottom (W): two piv_lo within look, second higher, break neckline
    last_pl_price = lo.shift(1).where(piv_lo).ffill()
    last_pl_i = pd.Series(np.arange(len(ltf)), index=ltf.index).where(piv_lo).ffill()
    idx = pd.Series(np.arange(len(ltf)), index=ltf.index)
    dist = idx - last_pl_i
    # second low near first (within 1% of price) and higher low
    near = (lo.shift(1) - last_pl_price).abs() / last_pl_price.replace(0, np.nan) < 0.003
    higher_low = lo.shift(1) > last_pl_price
    w_setup = piv_lo & (dist > 3) & (dist < look) & near & higher_low
    # neckline approx: high between the two lows (use rolling max of last look)
    neck = hi.rolling(look).max()
    long_e = w_setup.shift(1).fillna(False) & (c > neck.shift(1))
    # double top (M)
    last_ph_price = hi.shift(1).where(piv_hi).ffill()
    last_ph_i = pd.Series(np.arange(len(ltf)), index=ltf.index).where(piv_hi).ffill()
    dist2 = idx - last_ph_i
    near2 = (hi.shift(1) - last_ph_price).abs() / last_ph_price.replace(0, np.nan) < 0.003
    lower_high = hi.shift(1) < last_ph_price
    m_setup = piv_hi & (dist2 > 3) & (dist2 < look) & near2 & lower_high
    floor = lo.rolling(look).min()
    short_e = m_setup.shift(1).fillna(False) & (c < floor.shift(1))
    sig = pd.Series(0.0, index=ltf.index)
    sig[long_e.fillna(False)] = 1.0
    sig[short_e.fillna(False)] = -1.0
    sticky = sig.replace(0.0, np.nan).ffill().fillna(0.0)
    return align_sig(ltf.index, m1.index, sticky)


def proxy_hs_pattern(m1: pd.DataFrame, tf: str = "30min", look: int = 40) -> np.ndarray:
    """Crude head-and-shoulders: three peaks, middle highest; break neckline."""
    ltf = resample_ohlc(m1, tf)
    hi, lo, c = ltf["high"], ltf["low"], ltf["close"]
    piv_hi = (hi.shift(1) > hi.shift(2)) & (hi.shift(1) > hi) & (hi.shift(1) > hi.shift(3))
    # store last 3 pivot highs prices
    ph = hi.shift(1).where(piv_hi)
    # rolling list approach: use three last non-nan
    ph_ff = ph.dropna()
    # too heavy to be perfect; use: recent max is head, shoulders lower on both sides of window
    roll_max = hi.rolling(look).max()
    # head near middle of window and higher than edges
    left = hi.shift(look // 2)
    right = hi
    mid = hi.shift(look // 4)
    is_head = (mid >= roll_max * 0.995) & (mid > left) & (mid > right)
    neck = lo.rolling(look).min()
    short_e = is_head.shift(2).fillna(False) & (c < neck)
    # inverse H&S
    roll_min = lo.rolling(look).min()
    midl = lo.shift(look // 4)
    leftl = lo.shift(look // 2)
    rightl = lo
    is_inv = (midl <= roll_min * 1.005) & (midl < leftl) & (midl < rightl)
    neck_up = hi.rolling(look).max()
    long_e = is_inv.shift(2).fillna(False) & (c > neck_up)
    sig = pd.Series(0.0, index=ltf.index)
    sig[long_e.fillna(False)] = 1.0
    sig[short_e.fillna(False)] = -1.0
    sticky = sig.replace(0.0, np.nan).ffill().fillna(0.0)
    return align_sig(ltf.index, m1.index, sticky)


def proxy_rsi_reversal(m1: pd.DataFrame, tf: str = "5min", period: int = 14) -> np.ndarray:
    """Generic reversal proxy: RSI OS/OB cross back (not academy SID)."""
    ltf = resample_ohlc(m1, tf)
    delta = ltf["close"].diff()
    up = delta.clip(lower=0).rolling(period).mean()
    dn = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / dn.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    long_e = (rsi.shift(1) < 30) & (rsi >= 30)
    short_e = (rsi.shift(1) > 70) & (rsi <= 70)
    sig = pd.Series(0.0, index=ltf.index)
    sig[long_e.fillna(False)] = 1.0
    sig[short_e.fillna(False)] = -1.0
    sticky = sig.replace(0.0, np.nan).ffill().fillna(0.0)
    return align_sig(ltf.index, m1.index, sticky)


def proxy_sma_trend(m1: pd.DataFrame, tf: str = "15min") -> np.ndarray:
    """Baseline: SMA stack trend — already similar to existing agents."""
    ltf = resample_ohlc(m1, tf)
    s20 = ltf["close"].rolling(20).mean()
    s50 = ltf["close"].rolling(50).mean()
    sig = pd.Series(0.0, index=ltf.index)
    sig[s20 > s50] = 1.0
    sig[s20 < s50] = -1.0
    return align_sig(ltf.index, m1.index, sig)


def score_all_proxies(m1: pd.DataFrame) -> dict:
    close = m1["close"].to_numpy(dtype=np.float64)
    builders = {
        "proxy_bollinger_mr_5m": lambda: proxy_bollinger(m1, "5min"),
        "proxy_fib_pullback_15m": lambda: proxy_fib_pullback(m1, "15min"),
        "proxy_supply_demand_15m": lambda: proxy_supply_demand(m1, "15min"),
        "proxy_orderblock_15m": lambda: proxy_orderblock(m1, "15min"),
        "proxy_mw_15m": lambda: proxy_mw_pattern(m1, "15min"),
        "proxy_hs_30m": lambda: proxy_hs_pattern(m1, "30min"),
        "proxy_rsi_reversal_5m": lambda: proxy_rsi_reversal(m1, "5min"),
        "baseline_sma_trend_15m": lambda: proxy_sma_trend(m1, "15min"),
    }
    out = {}
    for name, fn in builders.items():
        try:
            side = fn()
            # clamp weird
            side = np.sign(side)
            row = {"name": name}
            for h in HORIZONS:
                hr = hit_rate(side, close, h)
                row[f"n_fires_h{h}"] = hr["n_fires"]
                row[f"hit_{h}"] = hr["hit"]
            # pulse density: fraction non-zero
            row["nonzero_frac"] = round(float(np.mean(side != 0)), 4)
            out[name] = row
        except Exception as e:
            out[name] = {"name": name, "error": str(e)}
    return out


def verdicts(sheet: dict, proxy_by_symbol: dict) -> list[dict]:
    """Brutal per-strategy verdict for signals agent."""
    # average proxy hits across symbols at h=10
    def avg_hit(proxy_name: str) -> float | None:
        hits = []
        for sym, res in proxy_by_symbol.items():
            r = res.get(proxy_name) or {}
            if r.get("hit_10") is not None and (r.get("n_fires_h10") or 0) >= 50:
                hits.append(r["hit_10"])
        return round(float(np.mean(hits)), 4) if hits else None

    # map academy names -> proxies
    mapping = [
        {
            "strategy": "Bollinger Bands",
            "sheet_key": "Bollinger Bands",
            "proxy": "proxy_bollinger_mr_5m",
            "already_in_slots": "YES — slots 90-92 bb_rsi_sma + phase_bb_mid",
            "codable": True,
        },
        {
            "strategy": "Fibonacci",
            "sheet_key": "Fibonacci",
            "proxy": "proxy_fib_pullback_15m",
            "already_in_slots": "NO pure fib slot; trend pullback covered by rsi2/stoch/sma families",
            "codable": True,
        },
        {
            "strategy": "Supply & Demand",
            "sheet_key": "Supply & Demand",
            "proxy": "proxy_supply_demand_15m",
            "already_in_slots": "NO dedicated S/D zone agent",
            "codable": "weak",
        },
        {
            "strategy": "Orderblocks",
            "sheet_key": "Orderblocks",
            "proxy": "proxy_orderblock_15m",
            "already_in_slots": "NO dedicated OB agent",
            "codable": "weak",
        },
        {
            "strategy": "Reversal Method (M&W / H&S)",
            "sheet_key": "Reversal Method",
            "proxy": "proxy_mw_15m",
            "already_in_slots": "NO pattern agent; RSI/stoch reversals exist",
            "codable": "weak",
        },
        {
            "strategy": "Head & Shoulders (named)",
            "sheet_key": "Head & Shoulders",
            "proxy": "proxy_hs_30m",
            "already_in_slots": "NO",
            "codable": "weak",
        },
        {
            "strategy": "SID Method",
            "sheet_key": "SID Method",
            "proxy": "proxy_rsi_reversal_5m",
            "already_in_slots": "NO — proprietary discretionary; not published as rules",
            "codable": False,
        },
        {
            "strategy": "DXY",
            "sheet_key": "DXY",
            "proxy": None,
            "already_in_slots": "NO — external symbol correlation, not a price-pattern agent on trade pair",
            "codable": False,
        },
    ]

    results = []
    for m in mapping:
        ss = sheet["strategies"].get(m["sheet_key"], {})
        ph = avg_hit(m["proxy"]) if m["proxy"] else None
        # baseline
        base = avg_hit("baseline_sma_trend_15m")

        # decision rules (brutal):
        # - need measurable algorithmic edge on OUR data: hit_10 >= 0.53 with enough fires
        # - and not already covered
        # - and not uncodable proprietary
        add = "SKIP"
        why = []
        if m["strategy"].startswith("DXY"):
            add = "SKIP"
            why.append("Not a tradeable signal family on the pair; needs DXY feed + correlation model. Wrong abstraction for signal slots.")
        elif m["codable"] is False:
            add = "SKIP"
            why.append("No closed-form rules in the sheet. Cannot code SID/academy black box without reverse-engineering.")
            if ph is not None and ph < 0.53:
                why.append(f"Even generic reversal proxy hit10={ph} is coin-flip / sub-threshold.")
        elif m["already_in_slots"].startswith("YES"):
            add = "SKIP (already covered)"
            why.append("We already have BB family agents with measured recipes. Do not add a second weaker BB.")
            if ph is not None:
                why.append(f"Naive BB proxy hit10={ph} (baseline SMA hit10={base}).")
        else:
            if ph is None:
                add = "SKIP"
                why.append("Proxy failed or too few fires on our data.")
            elif ph < 0.52:
                add = "SKIP"
                why.append(f"Proxy hit10={ph} is noise / worse than coin. Marketing WR does not transfer.")
            elif ph < 0.55:
                add = "MAYBE later (research only)"
                why.append(f"Proxy hit10={ph} barely above noise. Not worth a slot until it beats agree/dvmr on score_signal_accuracy.")
            else:
                add = "CANDIDATE"
                why.append(f"Proxy hit10={ph} looks usable. Still must beat slots 80-83 / bb_rsi_sma on prove path before enabling.")

        # sheet claim honesty
        if ss:
            why.append(
                f"Sheet: n={ss.get('n')} students, WR med={ss.get('wr_med')}%, "
                f"ret med={ss.get('ret_med')}% — ALL selected winners, no losers published."
            )

        results.append(
            {
                "strategy": m["strategy"],
                "verdict": add,
                "sheet_n": ss.get("n"),
                "sheet_wr_med": ss.get("wr_med"),
                "sheet_ret_med": ss.get("ret_med"),
                "our_proxy": m["proxy"],
                "our_hit10": ph,
                "baseline_sma_hit10": base,
                "already": m["already_in_slots"],
                "why": why,
            }
        )
    return results


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_sheet()
    sheet = sheet_audit(rows)

    print("=" * 72)
    print("STUDENT STRATEGY SHEET AUDIT (brutal)")
    print("=" * 72)
    print(f"rows={sheet['n_rows']} strategies={sheet['n_strategies']} dup_groups={sheet['n_duplicate_base_strat_groups']}")
    print(f"all returns positive? {sheet['all_published_returns_positive']}  min_ret={sheet['min_ret']} max_ret={sheet['max_ret']}")
    print("missing fields: trade_count, max_dd, sample_size, R-multiple, OOS split")
    print()
    print(f"{'Strategy':22} {'n':>4} {'WRmed':>6} {'WRstd':>6} {'>=70%':>6} {'RETmed':>7}")
    for s, stt in sorted(sheet["strategies"].items(), key=lambda kv: -kv[1]["n"]):
        print(
            f"{s:22} {stt['n']:4d} {stt['wr_med']:6.1f} {stt['wr_std']:6.1f} "
            f"{stt['frac_wr_ge_70']*100:5.1f}% {stt['ret_med']:7.1f}"
        )
    print()
    for rf in sheet["red_flags"]:
        print(f"  FLAG: {rf}")

    proxy_by_symbol = {}
    print()
    print("=" * 72)
    print("HONEST PROXIES ON OUR DATA (direction hit @ 5/10/20 M1 bars)")
    print("These are NOT the academy rules — only codable approximations.")
    print("=" * 72)
    for path in DATA_CANDIDATES:
        m1 = load_m1(path)
        if m1 is None or len(m1) < 5000:
            print(f"  skip {path.name}: missing or short")
            continue
        sym = path.stem.split("_")[0]
        print(f"\n--- {sym} bars={len(m1):,} {m1.index[0]} -> {m1.index[-1]} ---")
        res = score_all_proxies(m1)
        proxy_by_symbol[sym] = res
        for name, r in res.items():
            if "error" in r:
                print(f"  {name:30} ERROR {r['error']}")
                continue
            print(
                f"  {name:30} nz={r['nonzero_frac']:.3f} "
                f"hit5={r['hit_5']} (n={r['n_fires_h5']}) "
                f"hit10={r['hit_10']} (n={r['n_fires_h10']}) "
                f"hit20={r['hit_20']}"
            )

    v = verdicts(sheet, proxy_by_symbol)
    print()
    print("=" * 72)
    print("VERDICTS — worth adding to signals agent?")
    print("=" * 72)
    for item in v:
        print(f"\n[{item['verdict']}] {item['strategy']}")
        print(f"  sheet: n={item['sheet_n']} WRmed={item['sheet_wr_med']} RETmed={item['sheet_ret_med']}")
        print(f"  our hit10={item['our_hit10']}  baseline_sma hit10={item['baseline_sma_hit10']}")
        print(f"  already: {item['already']}")
        for w in item["why"]:
            print(f"  - {w}")

    report = {
        "sheet": sheet,
        "proxies": proxy_by_symbol,
        "verdicts": v,
        "bottom_line": {
            "add_now": [x["strategy"] for x in v if x["verdict"] == "CANDIDATE"],
            "maybe_research": [x["strategy"] for x in v if "MAYBE" in x["verdict"]],
            "skip": [x["strategy"] for x in v if x["verdict"].startswith("SKIP")],
            "note": (
                "Student sheet cannot be taken as evidence for signal slots. "
                "Our mission filter: raise clear% or protect breach 0 on prove_it. "
                "None of these marketing WRs prove that without closed rules + our scoreboard."
            ),
        },
    }
    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
