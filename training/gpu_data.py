"""Per-day tensor prep for the GPU Edition (Bot 1.5) — the shared-truth bridge.

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty (Bot 1.5 GPU Edition, 2026-07-20).
WHAT:  Turns an MT5 M1 CSV into two aligned per-day arrays the FastSim twin
       consumes, using the SAME loader + feature engine as the real bot so
       the brain sees byte-identical inputs (no mismatch):
         days_obs  (D, Lmax, C)  the brain's C market-feature columns
                                  (features.engine.obs_columns order — EXACT)
         days_phys (D, Lmax, 7)  raw physics the sim needs:
                                  high, low, close, spread(points), 15min ATR14,
                                  mask_buy_blocked, mask_sell_blocked
       + day_lens (D,), dates (D,), and the obs column list.
WHEN:  2026-07-20.
WHERE: consumed by training/fastsim.py and scripts/gpu_train.py.
WHY:   The slow pandas feature build must run ONCE; training then reads
       tensors. Using obs_columns() (not a hardcoded list) guarantees the
       column ORDER matches what PROVEN_2x was trained on.
INTERCONNECTED WITH: data_io/loader, features/engine, training/fastsim.
----------------------------------------------------------------------

CHANGE LOG (newest first — APPEND on every edit with date + WHY; keep this line):
- 2026-07-25  cache plain-English comments + load_multi_symbol_pool — WHY: SIGON Colab path.
- 2026-07-20  created — WHY: Bot 1.5 GPU Edition needs cached per-day tensors
  built by the real feature engine so the twin feeds the brain identical obs.
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
from __future__ import annotations
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.configs import path as rpath                          # noqa: E402
from data_io.loader import read_mt5_m1, trading_days            # noqa: E402
from features.engine import build_features, obs_columns         # noqa: E402

# ── CACHE (plain English) ──────────────────────────────────────────
# A CACHE is a saved shortcut of pre-built features (npz files under
# artifacts/gpu_cache_*.npz and artifacts/symbol_cache/*).
# After you turn signal slots ON/OFF or add new symbols, DELETE those
# cache files ONCE, then rebuild. NEVER delete .pt brain checkpoints
# when clearing caches.
# ───────────────────────────────────────────────────────────────────

# Raw columns the SIM needs (not the brain). spread is in POINTS here; the sim
# multiplies by POINT_SIZE. 15min::atr14 is the LTF ATR the broker stop uses.
PHYS_COLS = ["high", "low", "close", "spread",
             "15min::atr14", "mask_buy_blocked", "mask_sell_blocked"]


def build_day_tensors(csv_path: str, cache_path: str | None = None,
                      min_bars: int = 300, verbose: bool = True):
    """Return (days_obs, days_phys, day_lens, dates, cols). Cache to .npz.

    Cache = shortcut of features. If signals/columns changed, delete the .npz first.
    """
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        if verbose:
            print("gpu_data: loaded cache %s | days=%d Lmax=%d cols=%d"
                  % (cache_path, z["days_obs"].shape[0], z["days_obs"].shape[1],
                     z["days_obs"].shape[2]), flush=True)
            print("gpu_data: (cache is a feature shortcut — delete gpu_cache_*.npz / "
                  "symbol_cache if you flipped signals ON)", flush=True)
        return (z["days_obs"], z["days_phys"], z["day_lens"],
                list(z["dates"]), list(z["cols"]))

    if verbose:
        print("gpu_data: building features from %s (one-time)..." % csv_path, flush=True)
    m1 = read_mt5_m1(csv_path)
    F = build_features(m1)
    cols = obs_columns(F)                      # EXACT order the brain trained on
    days = trading_days(F)                      # [(date, F_day)], already >=300 filtered
    days = [(d, g) for d, g in days if len(g) >= min_bars]
    D = len(days)
    Lmax = max(len(g) for _, g in days)
    C = len(cols)

    days_obs = np.zeros((D, Lmax, C), dtype=np.float32)
    days_phys = np.zeros((D, Lmax, len(PHYS_COLS)), dtype=np.float32)
    day_lens = np.zeros(D, dtype=np.int64)
    dates = []
    for i, (date, g) in enumerate(days):
        n = len(g)
        # obs matrix — match env._obs_matrix exactly (nan->0, +inf->5, -inf->-5)
        M = g[cols].to_numpy(dtype=np.float32)
        M = np.nan_to_num(M, nan=0.0, posinf=5.0, neginf=-5.0)
        days_obs[i, :n] = M
        # physics — nan->0 so ATR-warmup rows reject opens (atr<=0), like DaySim
        P = g[PHYS_COLS].to_numpy(dtype=np.float32)
        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
        days_phys[i, :n] = P
        day_lens[i] = n
        dates.append(str(date))

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, days_obs=days_obs, days_phys=days_phys,
                            day_lens=day_lens, dates=np.array(dates, dtype=object),
                            cols=np.array(cols, dtype=object))
        if verbose:
            print("gpu_data: cached -> %s" % cache_path, flush=True)

    if verbose:
        print("gpu_data: DONE | days=%d Lmax=%d cols=%d | obs_dim would be %d"
              % (D, Lmax, C, 10 * (C + 12)), flush=True)
    return days_obs, days_phys, day_lens, dates, cols


# per-symbol physics constants (MT5 typical). Data-prep records these so the sim can use the
# right spread/size scale per market later. Obs features are normalized -> symbol-agnostic,
# so ONE brain handles every symbol.
SYMBOL_SPECS = {
    "XAUUSD": {"point_size": 0.01, "contract_size": 100.0},
    "EURUSD": {"point_size": 0.0001, "contract_size": 100000.0},
    "GBPUSD": {"point_size": 0.0001, "contract_size": 100000.0},
    "US30":   {"point_size": 0.1, "contract_size": 1.0},
}


def _symbol_of(path: str) -> str:
    name = os.path.basename(path).upper()
    for s in SYMBOL_SPECS:
        if s in name:
            return s
    return os.path.splitext(os.path.basename(path))[0].upper()


def build_symbol_set(csv_dir: str, cache_dir: str | None = None,
                     symbols=None, verbose: bool = True) -> dict:
    """Build + cache per-day tensors for EVERY MT5 M1 CSV in csv_dir (one file per symbol —
    e.g. the mounted Drive folder Camillion_data). One-time, cached per symbol. Returns
    {SYMBOL: {'cache','specs','days','cols'}}. Obs columns are identical across symbols, so
    one brain trains on them all."""
    import glob as _glob
    cache_dir = cache_dir or rpath("artifacts", "symbol_cache")
    os.makedirs(cache_dir, exist_ok=True)
    files = sorted(_glob.glob(os.path.join(csv_dir, "*.csv")))
    out = {}
    for f in files:
        sym = _symbol_of(f)
        if symbols and sym not in symbols:
            continue
        cache = os.path.join(cache_dir, "days_%s.npz" % sym)
        do, dp, dl, dates, cols = build_day_tensors(f, cache_path=cache, verbose=verbose)
        out[sym] = {"cache": cache,
                    "specs": SYMBOL_SPECS.get(sym, {"point_size": 0.01, "contract_size": 100.0}),
                    "days": int(do.shape[0]), "cols": int(do.shape[2])}
        if verbose:
            print("symbol %-8s | %d days | cols %d | -> %s"
                  % (sym, do.shape[0], do.shape[2], os.path.basename(cache)), flush=True)
    if not out:
        print("build_symbol_set: no CSVs found in %s" % csv_dir, flush=True)
    return out


def _find_symbol_csv(csv_dir: str, symbol: str) -> str | None:
    """Prefer curriculum-style CSVs, then any *SYMBOL*.csv."""
    import glob as _glob
    sym = symbol.upper()
    patterns = [
        os.path.join(csv_dir, f"{sym}_curriculum*.csv"),
        os.path.join(csv_dir, f"{sym}_M1*.csv"),
        os.path.join(csv_dir, f"{sym}*.csv"),
        os.path.join(csv_dir, f"*{sym}*.csv"),
    ]
    for pat in patterns:
        hits = sorted(_glob.glob(pat))
        # Prefer non-drill curriculum when multiple
        preferred = [h for h in hits if "drill" not in os.path.basename(h).lower()]
        pool = preferred or hits
        if pool:
            return pool[0]
    return None


def load_multi_symbol_pool(
    csv_dir: str,
    symbols: list[str] | None = None,
    cache_dir: str | None = None,
    min_bars: int = 300,
    verbose: bool = True,
):
    """Load and concatenate per-day tensors for multiple symbols.

    Returns
    -------
    days_obs, days_phys, day_lens, dates, cols, symbol_names
        symbol_names is a list length D (one label per day row).
    Missing symbols are skipped with a warning (XAUUSD-only still trains).
    """
    symbols = [s.strip().upper() for s in (symbols or ["XAUUSD"]) if s and s.strip()]
    cache_dir = cache_dir or rpath("artifacts", "symbol_cache")
    os.makedirs(cache_dir, exist_ok=True)

    blocks = []  # (do, dp, dl, dates, cols, sym)
    for sym in symbols:
        path = _find_symbol_csv(csv_dir, sym)
        if path is None:
            if verbose:
                print("load_multi_symbol_pool: SKIP %s — no CSV in %s" % (sym, csv_dir), flush=True)
            continue
        cache = os.path.join(cache_dir, "days_%s.npz" % sym)
        if verbose:
            print("load_multi_symbol_pool: %s <- %s" % (sym, path), flush=True)
        do, dp, dl, dates, cols = build_day_tensors(
            path, cache_path=cache, min_bars=min_bars, verbose=verbose
        )
        blocks.append((do, dp, dl, dates, cols, sym))

    if not blocks:
        # Hard fallback: classic gold curriculum
        src = rpath("data", "XAUUSD_curriculum_2026.csv")
        if verbose:
            print("load_multi_symbol_pool: no symbol CSVs — fallback %s" % src, flush=True)
        cache = rpath("artifacts", "gpu_cache_XAUUSD_curriculum_2026.npz")
        do, dp, dl, dates, cols = build_day_tensors(src, cache_path=cache, verbose=verbose)
        return do, dp, dl, dates, cols, ["XAUUSD"] * int(do.shape[0])

    # Column order must match across symbols (same feature engine)
    ref_cols = list(blocks[0][4])
    for _, _, _, _, cols, sym in blocks[1:]:
        if list(cols) != ref_cols:
            raise RuntimeError(
                "obs column mismatch for %s vs first symbol — rebuild caches after feature change"
                % sym
            )

    # Pad Lmax to global max, concat on day axis
    Lmax = max(int(b[0].shape[1]) for b in blocks)
    C = len(ref_cols)
    P = int(blocks[0][1].shape[2])

    obs_parts, phys_parts, lens_parts, date_parts, sym_parts = [], [], [], [], []
    for do, dp, dl, dates, cols, sym in blocks:
        D, L, _ = do.shape
        if L < Lmax:
            pad_o = np.zeros((D, Lmax - L, C), dtype=np.float32)
            pad_p = np.zeros((D, Lmax - L, P), dtype=np.float32)
            do = np.concatenate([do, pad_o], axis=1)
            dp = np.concatenate([dp, pad_p], axis=1)
        elif L > Lmax:
            do, dp = do[:, :Lmax], dp[:, :Lmax]
        obs_parts.append(do)
        phys_parts.append(dp)
        lens_parts.append(dl.astype(np.int64))
        date_parts.extend(list(dates))
        sym_parts.extend([sym] * D)

    days_obs = np.concatenate(obs_parts, axis=0)
    days_phys = np.concatenate(phys_parts, axis=0)
    day_lens = np.concatenate(lens_parts, axis=0)
    if verbose:
        from collections import Counter
        print(
            "load_multi_symbol_pool: DONE | days=%d Lmax=%d cols=%d | by_symbol=%s | obs_dim~%d"
            % (days_obs.shape[0], Lmax, C, dict(Counter(sym_parts)), 10 * (C + 12)),
            flush=True,
        )
    return days_obs, days_phys, day_lens, date_parts, ref_cols, sym_parts


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else rpath("data", "XAUUSD_curriculum_2026.csv")
    tag = os.path.splitext(os.path.basename(src))[0]
    cache = rpath("artifacts", "gpu_cache_%s.npz" % tag)
    do, dp, dl, dates, cols = build_day_tensors(src, cache_path=cache)
    print("shapes:", do.shape, dp.shape, dl.shape)
    print("day_lens: min %d max %d mean %.0f" % (dl.min(), dl.max(), dl.mean()))
    print("first/last date:", dates[0], "->", dates[-1], "| %d days" % len(dates))
    # sanity: close prices gold-ish, masks binary, some ATR positive
    import numpy as _np
    close = dp[..., 2][dp[..., 2] > 0]
    atr = dp[..., 4]
    mb = dp[..., 5]
    print("close range: %.1f .. %.1f" % (close.min(), close.max()))
    print("atr>0 fraction: %.2f | mask_buy in {0,1}: %s"
          % ((atr > 0).mean(), set(_np.unique(mb).tolist()) <= {0.0, 1.0}))
