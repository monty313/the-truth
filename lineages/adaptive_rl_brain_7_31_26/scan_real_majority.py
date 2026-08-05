"""Scan real data/raw for 92-agent majority frequency (plain report)."""
from __future__ import annotations

from lineages.adaptive_rl_brain_7_31_26.price_data import load_recent_bars, resolve_raw_csv
from lineages.adaptive_rl_brain_7_31_26.signal_majority import (
    compute_panel_matrix,
    majority_at,
)

FILES = [
    ("XAUUSD_M1_drill.csv", 8000),
    ("XAUUSD_curriculum_2026.csv", 8000),
    # large files: use first N rows only (read_mt5_m1 max_rows) — still real prices
    ("EURUSD_M1_curriculum.csv", 6000),
    ("US30_M1_curriculum.csv", 6000),
]


def scan(name: str, n_bars: int = 8000, step: int = 30) -> None:
    from lineages.adaptive_rl_brain_7_31_26.price_data import load_raw_m1

    try:
        p = resolve_raw_csv(name)
    except FileNotFoundError:
        print(f"skip missing {name}")
        return
    print(f"\n=== {name} (~{n_bars} bars, step {step}) ===")
    # Prefer tail for smaller files; head (max_rows) for multi-million-row CSVs
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > 40:
        m1 = load_raw_m1(name, max_rows=int(n_bars) + 5)
    else:
        m1 = load_recent_bars(n_bars, name=name)
    print(f"loaded bars={len(m1)} path={p} size_mb={size_mb:.1f}")
    mat, names = compute_panel_matrix(m1, only_enabled=False)
    ever = int((mat != 0).any(axis=0).sum())
    print(f"panel={mat.shape[1]} agents_ever_nonzero={ever}")
    maj = 0
    max_b = max_e = max_a = 0
    max_agree = 0.0
    samples = 0
    for t in range(720, len(m1), step):
        s = majority_at(mat, t)  # ≥10 active, ≥60% of active agree
        samples += 1
        max_b = max(max_b, s.n_bull)
        max_e = max(max_e, s.n_bear)
        max_a = max(max_a, s.n_active)
        if s.n_active:
            max_agree = max(max_agree, s.agree_frac)
        if s.has_majority:
            maj += 1
    print(
        f"samples={samples} consensus_hits={maj} "
        f"max_bull={max_b} max_bear={max_e} max_active={max_a} "
        f"max_agree_frac={max_agree:.2f} (need active>=10 and agree>=0.60)"
    )


if __name__ == "__main__":
    for f, n in FILES:
        scan(f, n_bars=n)
    print("\ndone")
