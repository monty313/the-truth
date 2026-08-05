"""Load real M1 price CSVs from repo data/raw for lineage sandbox tests.

CHANGE LOG:
- 2026-07-31  real price loader — WHY: majority / day tests should use Monty's
  folder data (data/raw/*.csv), not only synthetic thrust. Lineage only.

Preferred files (smallest first for fast tests):
  data/raw/XAUUSD_M1_drill.csv          ~40k bars
  data/raw/XAUUSD_curriculum_2026.csv   ~121k bars
  data/raw/EURUSD_M1_curriculum.csv     large
  data/raw/US30_M1_curriculum.csv       large
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

_LINEAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _LINEAGE_DIR.parents[1]
RAW_DIR = _REPO_ROOT / "data" / "raw"

# Prefer small → large so tests stay snappy
DEFAULT_CANDIDATES = (
    "XAUUSD_M1_drill.csv",
    "XAUUSD_curriculum_2026.csv",
    "EURUSD_M1_curriculum.csv",
    "US30_M1_curriculum.csv",
    "GBPUSD_M1_curriculum.csv",
    "XAUUSD_M1_full.csv",
)


def list_raw_csvs() -> List[Path]:
    if not RAW_DIR.is_dir():
        return []
    return sorted(RAW_DIR.glob("*.csv"))


def resolve_raw_csv(name: str | None = None) -> Path:
    """Resolve a CSV under data/raw. If name is None, pick first existing candidate."""
    if name:
        p = RAW_DIR / name
        if not p.is_file():
            raise FileNotFoundError(f"price file not found: {p}")
        return p
    for cand in DEFAULT_CANDIDATES:
        p = RAW_DIR / cand
        if p.is_file():
            return p
    found = list_raw_csvs()
    if not found:
        raise FileNotFoundError(f"no CSV price files under {RAW_DIR}")
    return found[0]


def load_raw_m1(
    name: str | None = None,
    *,
    max_rows: int | None = None,
    path: Path | str | None = None,
) -> pd.DataFrame:
    """Load MT5 M1 export via data_io.loader.read_mt5_m1."""
    from data_io.loader import read_mt5_m1

    p = Path(path) if path is not None else resolve_raw_csv(name)
    return read_mt5_m1(str(p), max_rows=max_rows)


def load_recent_bars(
    n_bars: int = 3000,
    *,
    name: str | None = None,
    path: Path | str | None = None,
) -> pd.DataFrame:
    """Load file and keep the last n_bars (enough for warm multi-TF + signals)."""
    m1 = load_raw_m1(name, path=path)
    if len(m1) <= n_bars:
        return m1
    return m1.iloc[-int(n_bars) :].copy()


def load_trading_days(
    n_days: int = 3,
    *,
    name: str | None = None,
    min_bars: int = 300,
    path: Path | str | None = None,
    prefer_recent: bool = True,
) -> List[Tuple[str, pd.DataFrame]]:
    """Return up to n_days full trading days from real CSV (date, day_frame)."""
    from data_io.loader import trading_days

    m1 = load_raw_m1(name, path=path)
    days = trading_days(m1)
    days = [(d, g) for d, g in days if len(g) >= min_bars]
    if prefer_recent:
        days = days[-int(n_days) :]
    else:
        days = days[: int(n_days)]
    return days


def data_banner() -> str:
    """One-line summary of what price file will be used."""
    try:
        p = resolve_raw_csv()
        return f"price_data={p.name} ({p})"
    except FileNotFoundError as e:
        return f"price_data=MISSING ({e})"
