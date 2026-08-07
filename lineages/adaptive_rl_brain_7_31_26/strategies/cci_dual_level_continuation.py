"""KAG strategy: dual CCI(30)+CCI(100) level-cross CONTINUATION (all official sets).

CHANGE LOG:
- 2026-08-06  created — WHY: First offline teacher for discover_physics_loop.
  Labels topology=continuation_release from rule; does NOT inject strategy
  flags into the 168-dim obs. Student must find links in existing board.

Rule (First Mark / KAG):
  BUY  continuation: both CCI30 and CCI100 **cross up through +100 on the LTF**,
       and at least one HTF has both CCI30+CCI100 **already through +100**
       (mass confirm — both ≥ +100 on that HTF bar).
  SELL continuation: mirror at **-100** (LTF dual-enter below; one HTF both ≤ -100).

LTF "cross" = dual-above (or dual-below) state **becomes true** on this bar.
HTF "through level" = both series on the extreme side on the as-of HTF bar
(simultaneous HTF enter is optional via strict_htf_cross=True).

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

from lineages.adaptive_rl_brain_7_31_26.data.mtf import bar_asof
from lineages.adaptive_rl_brain_7_31_26.perception.sets import OFFICIAL_SETS
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    TOPO_CHOP,
    TOPO_CONTINUATION,
    WAIT_NO_TRADE,
)

STRATEGY_ID = "cci_dual_level_continuation_v1"
STRATEGY_FAMILY = "CCI"
BULL_LEVEL = 100.0
BEAR_LEVEL = -100.0
MIN_BARS_CCI = 105  # CCI100 warm + 1
# ALWAYS all four official sets unless caller overrides with a non-empty list
ALL_SET_IDS = (1, 2, 3, 4)


@dataclass(frozen=True)
class ContinuationHit:
    set_id: int
    side: str  # "bull" | "bear"
    ltf: str
    htf_confirm: str
    action: int
    topology: int
    wait_subtype: int
    level: float


def dual_level_enter_mask(
    cci30: pd.Series,
    cci100: pd.Series,
    *,
    level: float,
    side: str,
) -> pd.Series:
    """True on bars where both CCIs newly enter the extreme side of `level`."""
    c30 = pd.to_numeric(cci30, errors="coerce")
    c100 = pd.to_numeric(cci100, errors="coerce")
    if side == "bull":
        both = (c30 >= level) & (c100 >= level)
    elif side == "bear":
        both = (c30 <= level) & (c100 <= level)
    else:
        raise ValueError(f"side must be bull|bear, got {side!r}")
    both = both.fillna(False).astype(bool)
    # fillna(False) alone can leave object dtype → ~ becomes bitwise -1/-2
    prev = both.shift(1)
    prev = prev.where(prev.notna(), False).astype(bool)
    return both & ~prev


def _pair_at(
    ind_tf: pd.DataFrame,
    ts: pd.Timestamp,
) -> Optional[Tuple[float, float, float, float]]:
    """Return (c30, c100, p30, p100) at as-of bar, or None if warm/invalid."""
    if ind_tf is None or len(ind_tf) < MIN_BARS_CCI:
        return None
    if "cci30" not in ind_tf.columns or "cci100" not in ind_tf.columns:
        return None
    i = bar_asof(ind_tf, ts)
    if i < MIN_BARS_CCI:
        return None
    c30 = float(ind_tf["cci30"].iloc[i])
    c100 = float(ind_tf["cci100"].iloc[i])
    p30 = float(ind_tf["cci30"].iloc[i - 1])
    p100 = float(ind_tf["cci100"].iloc[i - 1])
    if not all(np.isfinite(x) for x in (c30, c100, p30, p100)):
        return None
    return c30, c100, p30, p100


def _dual_active(c30: float, c100: float, *, side: str, level: float) -> bool:
    if side == "bull":
        return (c30 >= level) and (c100 >= level)
    return (c30 <= level) and (c100 <= level)


def _tf_enter_at(
    ind_tf: pd.DataFrame,
    ts: pd.Timestamp,
    *,
    side: str,
    level: float,
) -> bool:
    """True when dual-extreme state becomes true on this TF bar."""
    pair = _pair_at(ind_tf, ts)
    if pair is None:
        return False
    c30, c100, p30, p100 = pair
    now = _dual_active(c30, c100, side=side, level=level)
    prev = _dual_active(p30, p100, side=side, level=level)
    return bool(now and not prev)


def _tf_active_at(
    ind_tf: pd.DataFrame,
    ts: pd.Timestamp,
    *,
    side: str,
    level: float,
) -> bool:
    """True when both CCIs are through the level on this TF (mass confirm)."""
    pair = _pair_at(ind_tf, ts)
    if pair is None:
        return False
    c30, c100, _, _ = pair
    return bool(_dual_active(c30, c100, side=side, level=level))


def detect_continuation_at(
    ind: Mapping[str, pd.DataFrame],
    ts: pd.Timestamp,
    *,
    set_ids: Sequence[int] | None = None,
    strict_htf_cross: bool = False,
) -> List[ContinuationHit]:
    """Return all official-set continuation hits at wall-clock `ts`.

    LTF always requires dual level **enter** (cross).
    HTF default: dual level **active** (both through); set strict_htf_cross
    to require HTF dual enter on the same wall-clock as well.
    """
    if set_ids is None:
        want = set(ALL_SET_IDS)
    else:
        want = {int(s) for s in set_ids} or set(ALL_SET_IDS)
    hits: List[ContinuationHit] = []
    for s in OFFICIAL_SETS:
        if s.set_id not in want:
            continue
        ltf = s.entry_tf
        htfs = list(s.confirmation_tfs)
        fr_ltf = ind.get(ltf)
        if fr_ltf is None:
            continue
        for side, level, action in (
            ("bull", BULL_LEVEL, ACTION_BUY),
            ("bear", BEAR_LEVEL, ACTION_SELL),
        ):
            if not _tf_enter_at(fr_ltf, ts, side=side, level=level):
                continue
            confirm_htf = None
            for htf in htfs:
                fr_h = ind.get(htf)
                if fr_h is None:
                    continue
                if strict_htf_cross:
                    ok = _tf_enter_at(fr_h, ts, side=side, level=level)
                else:
                    ok = _tf_active_at(fr_h, ts, side=side, level=level)
                if ok:
                    confirm_htf = htf
                    break
            if confirm_htf is None:
                continue
            hits.append(
                ContinuationHit(
                    set_id=int(s.set_id),
                    side=side,
                    ltf=ltf,
                    htf_confirm=confirm_htf,
                    action=int(action),
                    topology=TOPO_CONTINUATION,
                    wait_subtype=WAIT_NO_TRADE,
                    level=float(level),
                )
            )
    return hits


def kag_lesson_row(hit: ContinuationHit, *, day: str = "", bar_index: int = 0) -> Dict[str, Any]:
    """Teacher lesson for KAG bus (principle_application, not copy_answer)."""
    return {
        "schema": "mark.teacher.lesson.v1",
        "strategy_id": STRATEGY_ID,
        "family": STRATEGY_FAMILY,
        "principle_ids": [
            "mass_vs_velocity",
            "ltf_continuation_release",
            "dual_cci_level_cross",
            "htf_confirm_one_of_two",
        ],
        "topology": "continuation_release",
        "act": "fire_buy" if hit.action == ACTION_BUY else "fire_sell",
        "wait_subtype": "no_trade",
        "set_id": hit.set_id,
        "ltf": hit.ltf,
        "htf_confirm": hit.htf_confirm,
        "side": hit.side,
        "level": hit.level,
        "day": day,
        "bar_index": bar_index,
        "forward_note": (
            "same construction if sensor renamed; learn dual-level enter "
            "on LTF + one HTF — do not memorize calendar date"
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
) -> Any:
    """Prepend prior calendar M1 so HTF CCI(100) can warm (4h/1d need many bars)."""
    import pandas as pd

    lo = max(0, int(day_index) - int(history_days))
    hi = int(day_index) + 1
    frames = [all_days[i][1] for i in range(lo, hi)]
    m1 = pd.concat(frames, axis=0)
    # drop duplicate timestamps (overnight joins)
    m1 = m1[~m1.index.duplicated(keep="last")].sort_index()
    return m1


def collect_continuation_dataset(
    days: Sequence[Tuple[str, Any]],
    *,
    max_days: int = 50,
    decide_every: int = 25,
    full_obs: bool = True,
    neg_per_pos: float = 2.0,
    seed: int = 42,
    set_ids: Sequence[int] | None = None,
    target: float = 2.0,
    risk: float = 3.0,
    history_days: int = 25,
    light_obs: bool = True,
) -> Dict[str, Any]:
    """Walk curriculum days; emit (obs, act, topo, wait) for multi-head BC.

    Positive: dual-CCI continuation hits (all sets).
    Negative: random non-hit bars → HOLD / CHOP / no_trade.
    Obs = existing full board only — strategy bits never written into obs.

    history_days: prepend prior M1 days so CCI100 warms on HTF (required for
    sets 3–4 and HTF confirm on 30m/1h/4h/1d).

    light_obs: skip 92-agent panel for speed; still MARK_FULL_DIM with zeros
    in agent slots (set/doctrine channels intact — where CCI confluence lives).
    """
    from lineages.adaptive_rl_brain_7_31_26.day_runner import DayRunner
    from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay
    from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import (
        MARK_FULL_DIM,
        build_mark_full_obs,
        pack_doctrine,
        pack_self_state,
    )

    rng = np.random.default_rng(seed)
    xs: List[np.ndarray] = []
    y_act: List[int] = []
    y_topo: List[int] = []
    y_wait: List[int] = []
    meta_rows: List[Dict[str, Any]] = []
    n_pos = 0
    n_neg = 0
    hits_by_set = {1: 0, 2: 0, 3: 0, 4: 0}

    day_list = list(days)
    # Start after enough history when possible
    start_i = 0
    end_i = min(len(day_list), max(max_days, 1))
    # Prefer days that have history_days of warmup behind them
    if len(day_list) > history_days + 2:
        start_i = min(history_days, max(0, len(day_list) - max_days))
        end_i = min(len(day_list), start_i + max_days)

    for di in range(start_i, end_i):
        date_str, m1_day = day_list[di]
        m1 = _concat_history_m1(day_list, di, history_days=history_days)
        # Indicator engine on long pack; labels only on current calendar day
        runner = DayRunner(
            m1,
            decide_every=decide_every,
            eyes_mode="mark_doctrine",
            use_signal_majority=not light_obs,
        )
        day_start = m1_day.index[0]
        day_end = m1_day.index[-1]

        # Decision bars only on the target calendar day (history is warm-up only)
        day_t0 = int(runner.m1.index.searchsorted(day_start, side="left"))
        day_t1 = int(runner.m1.index.searchsorted(day_end, side="right")) - 1
        day_t0 = max(day_t0, MIN_BARS_CCI)  # safety
        day_indices = list(range(day_t0, max(day_t0, day_t1) + 1, max(1, int(decide_every))))

        if light_obs:
            # Channel-1 board only (sets carry CCI group confluence) + zeros
            for t in day_indices:
                ts = runner.m1.index[int(t)]
                if ts < day_start or ts > day_end:
                    continue
                hits = detect_continuation_at(runner.ind, ts, set_ids=set_ids)
                # observe via day_runner path (32-dim channel1) → pad to 168
                ch1 = runner.observe(int(t))
                obs = build_mark_full_obs(
                    ch1,
                    doctrine_vec=pack_doctrine(getattr(runner, "last_doctrine", None)),
                    self_vec=pack_self_state(session_phase=0.5),
                )
                if hits:
                    hit = sorted(hits, key=lambda h: (-h.set_id, h.side))[0]
                    xs.append(obs)
                    y_act.append(int(hit.action))
                    y_topo.append(int(hit.topology))
                    y_wait.append(int(hit.wait_subtype))
                    meta_rows.append(
                        {
                            **kag_lesson_row(hit, day=str(date_str), bar_index=int(t)),
                            "kind": "positive",
                        }
                    )
                    n_pos += 1
                    hits_by_set[int(hit.set_id)] = hits_by_set.get(int(hit.set_id), 0) + 1
                elif rng.random() < 0.12:
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
        else:
            day = GoalEquityDay(
                m1,
                target_pct=target,
                risk_pct=risk,
                date_str=str(date_str),
                decide_every=decide_every,
                eyes_mode="mark_doctrine",
                mark_soul=True,
                full_obs=full_obs,
            )
            runner = day.runner
            for t in day_indices:
                if day.banked or day.dead:
                    break
                ts = runner.m1.index[int(t)]
                if ts < day_start or ts > day_end:
                    day.step_action(int(t), ACTION_HOLD)
                    continue
                hits = detect_continuation_at(runner.ind, ts, set_ids=set_ids)
                obs = np.asarray(day.observe(int(t)), dtype=np.float32).reshape(-1)
                if hits:
                    hit = sorted(hits, key=lambda h: (-h.set_id, h.side))[0]
                    xs.append(obs)
                    y_act.append(int(hit.action))
                    y_topo.append(int(hit.topology))
                    y_wait.append(int(hit.wait_subtype))
                    meta_rows.append(
                        {
                            **kag_lesson_row(hit, day=str(date_str), bar_index=int(t)),
                            "kind": "positive",
                        }
                    )
                    n_pos += 1
                    hits_by_set[int(hit.set_id)] = hits_by_set.get(int(hit.set_id), 0) + 1
                elif rng.random() < 0.12:
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
                day.step_action(int(t), ACTION_HOLD)

    # Rebalance negatives if too many
    if n_pos > 0 and n_neg > int(neg_per_pos * n_pos):
        pos_idx = [i for i, m in enumerate(meta_rows) if m.get("kind") == "positive"]
        neg_idx = [i for i, m in enumerate(meta_rows) if m.get("kind") == "negative"]
        keep_neg = int(neg_per_pos * n_pos)
        chosen_neg = list(rng.choice(neg_idx, size=min(keep_neg, len(neg_idx)), replace=False))
        keep = set(pos_idx + chosen_neg)
        order = sorted(keep)
        xs = [xs[i] for i in order]
        y_act = [y_act[i] for i in order]
        y_topo = [y_topo[i] for i in order]
        y_wait = [y_wait[i] for i in order]
        meta_rows = [meta_rows[i] for i in order]
        n_neg = len(chosen_neg)

    obs_dim = MARK_FULL_DIM if full_obs else int(xs[0].shape[0] if xs else 32)
    if not xs:
        return {
            "X": np.zeros((0, obs_dim), np.float32),
            "y_act": np.zeros((0,), np.int64),
            "y_topology": np.zeros((0,), np.int64),
            "y_wait": np.zeros((0,), np.int64),
            "n": 0,
            "n_pos": 0,
            "n_neg": 0,
            "hits_by_set": hits_by_set,
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
        "hits_by_set": hits_by_set,
        "strategy_id": STRATEGY_ID,
        "meta": meta_rows,
        "history_days": history_days,
    }
