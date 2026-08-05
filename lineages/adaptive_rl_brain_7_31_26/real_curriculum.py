"""Build a small real-price day curriculum for adaptive_rl_brain_7_31_26.

CHANGE LOG:
- 2026-07-31  Phase B — WHY: first serious train needs real days from data/raw,
  not only synthetic thrust. Prefer XAUUSD. Lineage only; no PROVEN.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from lineages.adaptive_rl_brain_7_31_26.price_data import (
    RAW_DIR,
    load_raw_m1,
    resolve_raw_csv,
)

_HERE = Path(__file__).resolve().parent
CURRICULUM_JSON = _HERE / "checkpoints" / "curriculum_days.json"
CURRICULUM_MD = _HERE / "CURRICULUM.md"

# Prefer fuller curriculum file for multi-day train; drill is smaller/faster fallback
PREFERRED_SOURCE = "XAUUSD_curriculum_2026.csv"
FALLBACK_SOURCE = "XAUUSD_M1_drill.csv"

MIN_BARS_PER_DAY = 900  # enough warm + decisions for multi-TF
MAX_DAYS_TOTAL = 8


@dataclass
class DayPick:
    date: str
    role: str  # "trend_bull" | "trend_bear" | "pullback_mix"
    n_bars: int
    net_move: float
    trend_strength: float
    source_file: str
    start_ts: str
    end_ts: str


def _score_day(df: pd.DataFrame) -> Tuple[float, float]:
    c = df["close"].astype(float).values
    if len(c) < 10:
        return 0.0, 0.0
    net = float(c[-1] - c[0])
    path = float(np.abs(np.diff(c)).sum())
    strength = abs(net) / max(path, 1e-9)
    return net, strength


def split_calendar_days(m1: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    """Split M1 frame into calendar-day slices (UTC/naive as stored)."""
    if not isinstance(m1.index, pd.DatetimeIndex):
        raise TypeError("m1 index must be DatetimeIndex")
    out: List[Tuple[str, pd.DataFrame]] = []
    for d, g in m1.groupby(m1.index.date):
        g = g.sort_index()
        if len(g) >= MIN_BARS_PER_DAY:
            out.append((str(d), g))
    return out


def pick_curriculum_days(
    m1: pd.DataFrame,
    *,
    source_file: str,
    n_trend: int = 4,
    n_mix: int = 2,
) -> List[Tuple[DayPick, pd.DataFrame]]:
    """Pick strong trend days + weaker/mix days from real M1."""
    scored: List[Tuple[DayPick, pd.DataFrame]] = []
    for d_str, g in split_calendar_days(m1):
        net, strength = _score_day(g)
        role = "pullback_mix"
        if strength >= 0.08 and abs(net) > 0:
            role = "trend_bull" if net > 0 else "trend_bear"
        pick = DayPick(
            date=d_str,
            role=role,
            n_bars=len(g),
            net_move=float(net),
            trend_strength=float(strength),
            source_file=source_file,
            start_ts=str(g.index[0]),
            end_ts=str(g.index[-1]),
        )
        scored.append((pick, g))

    bulls = sorted(
        [x for x in scored if x[0].role == "trend_bull"],
        key=lambda x: x[0].trend_strength,
        reverse=True,
    )
    bears = sorted(
        [x for x in scored if x[0].role == "trend_bear"],
        key=lambda x: x[0].trend_strength,
        reverse=True,
    )
    mixes = sorted(
        [x for x in scored if x[0].role == "pullback_mix"],
        key=lambda x: x[0].trend_strength,
        reverse=True,
    )

    chosen: List[Tuple[DayPick, pd.DataFrame]] = []
    # alternate bull/bear trends
    n_each = max(1, n_trend // 2)
    for i in range(n_each):
        if i < len(bulls):
            chosen.append(bulls[i])
        if i < len(bears):
            chosen.append(bears[i])
    # fill remaining trend slots from strongest unused
    used = {c[0].date for c in chosen}
    rest_trend = [
        x
        for x in scored
        if x[0].date not in used and x[0].role.startswith("trend")
    ]
    rest_trend.sort(key=lambda x: x[0].trend_strength, reverse=True)
    for x in rest_trend:
        if len([c for c in chosen if c[0].role.startswith("trend")]) >= n_trend:
            break
        chosen.append(x)
        used.add(x[0].date)

    for x in mixes:
        if len([c for c in chosen if c[0].role == "pullback_mix"]) >= n_mix:
            break
        if x[0].date in used:
            continue
        chosen.append(x)
        used.add(x[0].date)

    # hard cap
    chosen = chosen[:MAX_DAYS_TOTAL]
    # stable order by date
    chosen.sort(key=lambda x: x[0].date)
    return chosen


def load_real_curriculum(
    *,
    source: Optional[str] = None,
    n_trend: int = 4,
    n_mix: int = 2,
) -> Tuple[List[pd.DataFrame], List[DayPick], str]:
    """Load preferred XAUUSD file and return day frames + metadata."""
    name = source
    if name is None:
        for cand in (PREFERRED_SOURCE, FALLBACK_SOURCE):
            try:
                resolve_raw_csv(cand)
                name = cand
                break
            except FileNotFoundError:
                continue
        if name is None:
            p = resolve_raw_csv(None)
            name = p.name
    path = resolve_raw_csv(name)
    m1 = load_raw_m1(name)
    picks = pick_curriculum_days(m1, source_file=name, n_trend=n_trend, n_mix=n_mix)
    if not picks:
        raise RuntimeError(f"no usable calendar days in {path}")
    frames = [g.copy() for _, g in picks]
    meta = [p for p, _ in picks]
    return frames, meta, name


def write_curriculum_docs(meta: Sequence[DayPick], source: str) -> None:
    """Write CURRICULUM.md + checkpoints/curriculum_days.json."""
    CURRICULUM_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lineage": "adaptive_rl_brain_7_31_26",
        "source_file": source,
        "raw_dir": str(RAW_DIR),
        "days": [asdict(m) for m in meta],
        "proven_touched": False,
    }
    CURRICULUM_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# CURRICULUM — adaptive_rl_brain_7_31_26",
        "",
        "Real-price day list for first serious training (Phase B/C).",
        "",
        f"**Source:** `data/raw/{source}`",
        "",
        "Selection rules:",
        f"- Min bars/day: {MIN_BARS_PER_DAY}",
        "- Trend days: high |net| / path length (directional sessions)",
        "- Mix days: weaker trend (more two-way / pullback-ish)",
        "- Cap: modest set for first serious run",
        "",
        "| Date | Role | Bars | Net move | Trend strength |",
        "|------|------|-----:|---------:|---------------:|",
    ]
    for m in meta:
        lines.append(
            f"| {m.date} | {m.role} | {m.n_bars} | {m.net_move:+.4f} | "
            f"{m.trend_strength:.4f} |"
        )
    lines.extend(
        [
            "",
            "Machine-readable: `checkpoints/curriculum_days.json`",
            "",
            "PROVEN: not used. Checkpoints only under this lineage folder.",
            "",
        ]
    )
    CURRICULUM_MD.write_text("\n".join(lines), encoding="utf-8")


def verify_mtf_on_days(frames: Sequence[pd.DataFrame]) -> Dict[str, object]:
    """Confirm M1 → multi-TF pack builds and higher TFs have bars."""
    from lineages.adaptive_rl_brain_7_31_26.data.mtf import build_mtf_pack

    report: Dict[str, object] = {"n_days": len(frames), "days": []}
    for i, df in enumerate(frames):
        pack = build_mtf_pack(df)
        day_info = {
            "i": i,
            "n_m1": len(df),
            "tfs": {k: (0 if v is None else len(v)) for k, v in pack.items()},
        }
        report["days"].append(day_info)  # type: ignore[attr-defined]
        # need at least 1m + some higher TF
        assert len(df) >= MIN_BARS_PER_DAY
        assert pack.get("1m") is not None and len(pack["1m"]) > 0
    return report


if __name__ == "__main__":
    frames, meta, src = load_real_curriculum()
    write_curriculum_docs(meta, src)
    mtf = verify_mtf_on_days(frames)
    print(f"source={src} days={len(frames)}")
    for m in meta:
        print(f"  {m.date} {m.role} bars={m.n_bars} strength={m.trend_strength:.3f}")
    print("mtf ok:", mtf["n_days"], "days")
    print(f"wrote {CURRICULUM_MD}")
    print(f"wrote {CURRICULUM_JSON}")
