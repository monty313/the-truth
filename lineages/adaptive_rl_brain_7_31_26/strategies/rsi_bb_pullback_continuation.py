"""KAG strategy: RSI(5)+BB pullback / continuation (all official sets).

CHANGE LOG:
- 2026-08-06  created — WHY: Second offline teacher for discover_physics_loop.
  Labels pullback_load vs continuation_release from RSI-BB geometry.
  Does NOT inject strategy flags into 168-dim obs.

Rule (First Mark / KAG):
  LTF sensor: RSI(5) with Bollinger(period=10, deviation=0.5, shift=+5)
              applied **on the RSI series** (not on price).

  HTF mass filter: price vs Bollinger(period=100, deviation=0.5, shift=+2)
    BUY side:  close **above** BB mid on **both** HTFs of the set
    SELL side: close **below** BB mid on **both** HTFs of the set

  BUY pullback:      LTF RSI **below** its lower BB  → wait loaded (HOLD)
  BUY continuation:  LTF RSI **crosses up** its upper BB → fire BUY
  SELL pullback:     LTF RSI **above** its upper BB  → wait loaded (HOLD)
  SELL continuation: LTF RSI **crosses down** its lower BB → fire SELL

Sets (MARK SETS LAW):
  1: LTF 1m  | HTF 15m, 30m
  2: LTF 5m  | HTF 30m, 1h
  3: LTF 15m | HTF 1h, 4h
  4: LTF 30m | HTF 4h, 1d
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from features.indicators import bollinger, rsi
from lineages.adaptive_rl_brain_7_31_26.data.mtf import bar_asof
from lineages.adaptive_rl_brain_7_31_26.perception.sets import OFFICIAL_SETS
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    TOPO_CHOP,
    TOPO_CONTINUATION,
    TOPO_PULLBACK,
    WAIT_LOADED,
    WAIT_NO_TRADE,
)

STRATEGY_ID = "rsi_bb_pullback_continuation_v1"
STRATEGY_FAMILY = "RSI_BB"

# ALWAYS all four official sets (MARK SETS LAW) — never train on one stack only.
ALL_SET_IDS: Tuple[int, ...] = (1, 2, 3, 4)
SET_STACKS = {
    1: {"ltf": "1m", "htf": ("15m", "30m")},
    2: {"ltf": "5m", "htf": ("30m", "1h")},
    3: {"ltf": "15m", "htf": ("1h", "4h")},
    4: {"ltf": "30m", "htf": ("4h", "1d")},
}

# LTF RSI-BB
RSI_PERIOD = 5
RSI_BB_PERIOD = 10
RSI_BB_DEV = 0.5
RSI_BB_SHIFT = 5

# HTF price-BB mass filter
HTF_BB_PERIOD = 100
HTF_BB_DEV = 0.5
HTF_BB_SHIFT = 2

MIN_LTF_BARS = RSI_PERIOD + RSI_BB_PERIOD + RSI_BB_SHIFT + 2  # ~22
MIN_HTF_BARS = HTF_BB_PERIOD + HTF_BB_SHIFT + 2  # ~104


def resolve_set_ids(set_ids: Sequence[int] | None) -> Tuple[int, ...]:
    """Default = all four sets. Empty/None never silently shrinks the universe."""
    if set_ids is None:
        return ALL_SET_IDS
    got = tuple(sorted({int(s) for s in set_ids if int(s) in ALL_SET_IDS}))
    return got if got else ALL_SET_IDS


@dataclass(frozen=True)
class RsiBbHit:
    set_id: int
    side: str  # bull | bear
    kind: str  # pullback | continuation
    ltf: str
    htf_confirm: Tuple[str, str]
    action: int
    topology: int
    wait_subtype: int


def build_ltf_rsi_bb(ohlc: pd.DataFrame) -> pd.DataFrame:
    """RSI(5) + BB(10, 0.5, shift+5) on the RSI series."""
    r = rsi(ohlc["close"], RSI_PERIOD)
    up, mid, lo = bollinger(r, RSI_BB_PERIOD, RSI_BB_DEV, shift=RSI_BB_SHIFT)
    out = pd.DataFrame(index=ohlc.index)
    out["rsi5"] = r
    out["rsi_bb_up"] = up
    out["rsi_bb_mid"] = mid
    out["rsi_bb_lo"] = lo
    return out


def build_htf_price_bb(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Price BB(100, 0.5, shift+2)."""
    up, mid, lo = bollinger(ohlc["close"], HTF_BB_PERIOD, HTF_BB_DEV, shift=HTF_BB_SHIFT)
    out = pd.DataFrame(index=ohlc.index)
    out["close"] = ohlc["close"]
    out["px_bb_up"] = up
    out["px_bb_mid"] = mid
    out["px_bb_lo"] = lo
    return out


def precompute_rsi_bb_pack(pack: Mapping[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Per-TF frames: ltf_rsi_bb and htf_price_bb (same keys as pack)."""
    ltf_map: Dict[str, pd.DataFrame] = {}
    htf_map: Dict[str, pd.DataFrame] = {}
    for tf, df in pack.items():
        if df is None or len(df) == 0:
            continue
        ltf_map[tf] = build_ltf_rsi_bb(df)
        htf_map[tf] = build_htf_price_bb(df)
    return {"ltf_rsi_bb": ltf_map, "htf_price_bb": htf_map}


def _vals_at(frame: pd.DataFrame, ts: pd.Timestamp, cols: Sequence[str]) -> Optional[Dict[str, float]]:
    if frame is None or len(frame) < 2:
        return None
    i = bar_asof(frame, ts)
    if i < 1:
        return None
    out: Dict[str, float] = {"_i": float(i)}
    for c in cols:
        if c not in frame.columns:
            return None
        v = float(frame[c].iloc[i])
        if not np.isfinite(v):
            return None
        out[c] = v
        pv = float(frame[c].iloc[i - 1])
        out[f"{c}__prev"] = pv if np.isfinite(pv) else float("nan")
    return out


def htf_mass_ok(
    htf_bb: Mapping[str, pd.DataFrame],
    htfs: Sequence[str],
    ts: pd.Timestamp,
    *,
    side: str,
) -> bool:
    """Both HTFs: bull → close > mid BB; bear → close < mid BB."""
    for htf in htfs:
        fr = htf_bb.get(htf)
        if fr is None or len(fr) < MIN_HTF_BARS:
            return False
        i = bar_asof(fr, ts)
        if i < MIN_HTF_BARS:
            return False
        close = float(fr["close"].iloc[i])
        mid = float(fr["px_bb_mid"].iloc[i])
        if not np.isfinite(close) or not np.isfinite(mid):
            return False
        if side == "bull":
            if not (close > mid):
                return False
        else:
            if not (close < mid):
                return False
    return True


def detect_rsi_bb_at(
    pack_bb: Mapping[str, Mapping[str, pd.DataFrame]],
    ts: pd.Timestamp,
    *,
    set_ids: Sequence[int] | None = None,
) -> List[RsiBbHit]:
    """Return pullback/continuation hits on **every** requested official set.

    Default scans Sets 1–4 (same RSI-BB rule, different LTF/HTF stacks).
    """
    ltf_bb = pack_bb["ltf_rsi_bb"]
    htf_bb = pack_bb["htf_price_bb"]
    want = set(resolve_set_ids(set_ids))
    hits: List[RsiBbHit] = []

    for s in OFFICIAL_SETS:
        if s.set_id not in want:
            continue
        # Law stack must match SET_STACKS table
        law = SET_STACKS[int(s.set_id)]
        ltf = s.entry_tf
        htfs = tuple(s.confirmation_tfs)
        assert ltf == law["ltf"] and htfs == law["htf"], (
            f"set {s.set_id} stack drift: got {(ltf, htfs)} expected {law}"
        )
        fr = ltf_bb.get(ltf)
        if fr is None or len(fr) < MIN_LTF_BARS:
            continue
        i = bar_asof(fr, ts)
        if i < MIN_LTF_BARS:
            continue
        r = float(fr["rsi5"].iloc[i])
        up = float(fr["rsi_bb_up"].iloc[i])
        lo = float(fr["rsi_bb_lo"].iloc[i])
        r_p = float(fr["rsi5"].iloc[i - 1])
        up_p = float(fr["rsi_bb_up"].iloc[i - 1])
        lo_p = float(fr["rsi_bb_lo"].iloc[i - 1])
        if not all(np.isfinite(x) for x in (r, up, lo, r_p, up_p, lo_p)):
            continue

        # --- BUY side (HTF both above mid) ---
        if htf_mass_ok(htf_bb, htfs, ts, side="bull"):
            bull_cont = (r_p < up_p) and (r >= up)  # cross up upper band
            bull_pb = r < lo  # below lower band
            if bull_cont:
                hits.append(
                    RsiBbHit(
                        set_id=int(s.set_id),
                        side="bull",
                        kind="continuation",
                        ltf=ltf,
                        htf_confirm=htfs,
                        action=ACTION_BUY,
                        topology=TOPO_CONTINUATION,
                        wait_subtype=WAIT_NO_TRADE,
                    )
                )
            elif bull_pb:
                hits.append(
                    RsiBbHit(
                        set_id=int(s.set_id),
                        side="bull",
                        kind="pullback",
                        ltf=ltf,
                        htf_confirm=htfs,
                        action=ACTION_HOLD,
                        topology=TOPO_PULLBACK,
                        wait_subtype=WAIT_LOADED,
                    )
                )

        # --- SELL side (HTF both below mid) ---
        if htf_mass_ok(htf_bb, htfs, ts, side="bear"):
            bear_cont = (r_p > lo_p) and (r <= lo)  # cross down lower band
            bear_pb = r > up  # above upper band
            if bear_cont:
                hits.append(
                    RsiBbHit(
                        set_id=int(s.set_id),
                        side="bear",
                        kind="continuation",
                        ltf=ltf,
                        htf_confirm=htfs,
                        action=ACTION_SELL,
                        topology=TOPO_CONTINUATION,
                        wait_subtype=WAIT_NO_TRADE,
                    )
                )
            elif bear_pb:
                hits.append(
                    RsiBbHit(
                        set_id=int(s.set_id),
                        side="bear",
                        kind="pullback",
                        ltf=ltf,
                        htf_confirm=htfs,
                        action=ACTION_HOLD,
                        topology=TOPO_PULLBACK,
                        wait_subtype=WAIT_LOADED,
                    )
                )
    return hits


def kag_lesson_row(hit: RsiBbHit, *, day: str = "", bar_index: int = 0) -> Dict[str, Any]:
    topo_name = "continuation_release" if hit.kind == "continuation" else "pullback_load"
    if hit.kind == "continuation":
        act = "fire_buy" if hit.action == ACTION_BUY else "fire_sell"
        wait = "no_trade"
    else:
        act = "wait_loaded"
        wait = "loaded"
    return {
        "schema": "mark.teacher.lesson.v1",
        "strategy_id": STRATEGY_ID,
        "family": STRATEGY_FAMILY,
        "principle_ids": [
            "mass_vs_velocity",
            "ltf_rsi_bb_geometry",
            "htf_price_bb_mass",
            "pullback_load_then_release",
        ],
        "topology": topo_name,
        "act": act,
        "wait_subtype": wait,
        "set_id": hit.set_id,
        "ltf": hit.ltf,
        "htf_confirm": list(hit.htf_confirm),
        "side": hit.side,
        "kind": hit.kind,
        "day": day,
        "bar_index": bar_index,
        "params": {
            "rsi": RSI_PERIOD,
            "rsi_bb": {"period": RSI_BB_PERIOD, "dev": RSI_BB_DEV, "shift": RSI_BB_SHIFT},
            "htf_bb": {"period": HTF_BB_PERIOD, "dev": HTF_BB_DEV, "shift": HTF_BB_SHIFT},
            "htf_filter": "close_vs_mid_both_htfs",
        },
        "forward_note": (
            "same construction if RSI/BB family swapped; learn slingshot load "
            "vs release under HTF mass — do not memorize calendar date"
        ),
        "label_act": hit.action,
        "label_topology": hit.topology,
        "label_wait": hit.wait_subtype,
    }


def _concat_history_m1(
    all_days: Sequence[Tuple[str, Any]],
    day_index: int,
    *,
    history_days: int,
) -> pd.DataFrame:
    lo = max(0, int(day_index) - int(history_days))
    hi = int(day_index) + 1
    frames = [all_days[i][1] for i in range(lo, hi)]
    m1 = pd.concat(frames, axis=0)
    return m1[~m1.index.duplicated(keep="last")].sort_index()


def _pick_hit(hits: List[RsiBbHit]) -> RsiBbHit:
    """Prefer continuation over pullback; then higher set_id (macro mass)."""
    rank = {"continuation": 0, "pullback": 1}
    return sorted(hits, key=lambda h: (rank.get(h.kind, 9), -h.set_id, h.side))[0]


def collect_rsi_bb_dataset(
    days: Sequence[Tuple[str, Any]],
    *,
    max_days: int = 50,
    decide_every: int = 25,
    full_obs: bool = True,
    neg_per_pos: float = 2.0,
    seed: int = 42,
    set_ids: Sequence[int] | None = None,
    history_days: int = 120,
    light_obs: bool = True,
) -> Dict[str, Any]:
    """Walk curriculum; emit multi-head labels for RSI-BB pullback/continuation.

    Scans **all four official sets** by default (same rule on each stack).
    Obs unpoisoned (no strategy flags).

    history_days default **120**: Set 4 HTF includes **1d** BB(100) — needs
    ~100 daily bars of warm-up so set 4 is not silently empty.
    """
    from lineages.adaptive_rl_brain_7_31_26.day_runner import DayRunner
    from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import (
        MARK_FULL_DIM,
        build_mark_full_obs,
        pack_doctrine,
        pack_self_state,
    )

    active_sets = resolve_set_ids(set_ids)
    rng = np.random.default_rng(seed)
    xs: List[np.ndarray] = []
    y_act: List[int] = []
    y_topo: List[int] = []
    y_wait: List[int] = []
    meta_rows: List[Dict[str, Any]] = []
    n_pos = 0
    n_neg = 0
    n_pull = 0
    n_cont = 0
    hits_by_set = {sid: 0 for sid in ALL_SET_IDS}
    hits_by_set_kind = {
        sid: {"pullback": 0, "continuation": 0} for sid in ALL_SET_IDS
    }

    day_list = list(days)
    start_i = 0
    end_i = min(len(day_list), max(max_days, 1))
    if len(day_list) > history_days + 2:
        start_i = min(history_days, max(0, len(day_list) - max_days))
        end_i = min(len(day_list), start_i + max_days)

    for di in range(start_i, end_i):
        date_str, m1_day = day_list[di]
        m1 = _concat_history_m1(day_list, di, history_days=history_days)
        runner = DayRunner(
            m1,
            decide_every=decide_every,
            eyes_mode="mark_doctrine",
            use_signal_majority=not light_obs,
        )
        pack_bb = precompute_rsi_bb_pack(runner.pack)
        day_start = m1_day.index[0]
        day_end = m1_day.index[-1]
        day_t0 = int(runner.m1.index.searchsorted(day_start, side="left"))
        day_t1 = int(runner.m1.index.searchsorted(day_end, side="right")) - 1
        day_t0 = max(day_t0, MIN_LTF_BARS)
        day_indices = list(
            range(day_t0, max(day_t0, day_t1) + 1, max(1, int(decide_every)))
        )

        for t in day_indices:
            ts = runner.m1.index[int(t)]
            if ts < day_start or ts > day_end:
                continue
            hits = detect_rsi_bb_at(pack_bb, ts, set_ids=active_sets)
            ch1 = runner.observe(int(t))
            obs = build_mark_full_obs(
                ch1,
                doctrine_vec=pack_doctrine(getattr(runner, "last_doctrine", None)),
                self_vec=pack_self_state(session_phase=0.5),
            )
            if hits:
                # Count every set hit for coverage stats, label from best pick
                for h in hits:
                    hits_by_set[int(h.set_id)] = hits_by_set.get(int(h.set_id), 0) + 1
                    hits_by_set_kind[int(h.set_id)][h.kind] = (
                        hits_by_set_kind[int(h.set_id)].get(h.kind, 0) + 1
                    )
                hit = _pick_hit(hits)
                xs.append(obs)
                y_act.append(int(hit.action))
                y_topo.append(int(hit.topology))
                y_wait.append(int(hit.wait_subtype))
                meta_rows.append(
                    {
                        **kag_lesson_row(hit, day=str(date_str), bar_index=int(t)),
                        "kind": "positive",
                        "n_sets_hit_this_bar": len(hits),
                        "sets_hit": [h.set_id for h in hits],
                    }
                )
                n_pos += 1
                if hit.kind == "pullback":
                    n_pull += 1
                else:
                    n_cont += 1
            elif rng.random() < 0.10:
                xs.append(obs)
                y_act.append(ACTION_HOLD)
                y_topo.append(TOPO_CHOP)
                y_wait.append(WAIT_NO_TRADE)
                meta_rows.append(
                    {
                        "kind": "negative",
                        "day": str(date_str),
                        "bar_index": int(t),
                        "strategy_id": STRATEGY_ID,
                    }
                )
                n_neg += 1

    if n_pos > 0 and n_neg > int(neg_per_pos * n_pos):
        pos_idx = [i for i, m in enumerate(meta_rows) if m.get("kind") == "positive"]
        neg_idx = [i for i, m in enumerate(meta_rows) if m.get("kind") == "negative"]
        keep_neg = int(neg_per_pos * n_pos)
        chosen_neg = list(
            rng.choice(neg_idx, size=min(keep_neg, len(neg_idx)), replace=False)
        )
        keep = set(pos_idx + chosen_neg)
        order = sorted(keep)
        xs = [xs[i] for i in order]
        y_act = [y_act[i] for i in order]
        y_topo = [y_topo[i] for i in order]
        y_wait = [y_wait[i] for i in order]
        meta_rows = [meta_rows[i] for i in order]
        n_neg = len(chosen_neg)

    obs_dim = MARK_FULL_DIM if full_obs else int(xs[0].shape[0] if xs else 32)
    sets_with_hits = [sid for sid in ALL_SET_IDS if hits_by_set.get(sid, 0) > 0]
    if not xs:
        return {
            "X": np.zeros((0, obs_dim), np.float32),
            "y_act": np.zeros((0,), np.int64),
            "y_topology": np.zeros((0,), np.int64),
            "y_wait": np.zeros((0,), np.int64),
            "n": 0,
            "n_pos": 0,
            "n_neg": 0,
            "n_pullback": 0,
            "n_continuation": 0,
            "hits_by_set": hits_by_set,
            "hits_by_set_kind": hits_by_set_kind,
            "sets_active": list(active_sets),
            "sets_with_hits": sets_with_hits,
            "all_sets_covered": len(sets_with_hits) == len(ALL_SET_IDS),
            "set_stacks": SET_STACKS,
            "strategy_id": STRATEGY_ID,
            "meta": [],
            "history_days": history_days,
        }
    return {
        "X": np.stack(xs, axis=0).astype(np.float32),
        "y_act": np.asarray(y_act, dtype=np.int64),
        "y_topology": np.asarray(y_topo, dtype=np.int64),
        "y_wait": np.asarray(y_wait, dtype=np.int64),
        "n": int(len(y_act)),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "n_pullback": int(n_pull),
        "n_continuation": int(n_cont),
        "hits_by_set": hits_by_set,
        "hits_by_set_kind": hits_by_set_kind,
        "sets_active": list(active_sets),
        "sets_with_hits": sets_with_hits,
        "all_sets_covered": len(sets_with_hits) == len(ALL_SET_IDS),
        "set_stacks": SET_STACKS,
        "strategy_id": STRATEGY_ID,
        "meta": meta_rows,
        "history_days": history_days,
    }
