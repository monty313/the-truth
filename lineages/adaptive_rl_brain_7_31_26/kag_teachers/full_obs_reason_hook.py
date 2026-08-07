"""Hook: GoalEquityDay full_obs → Reason Teacher + learn-to-learn labels.

Use during BC collection so the clone SEE all 168 channels, gets multi-head
targets (act+topology+wait+roles), and pattern-transfer memory — not act-only.

Usage (repo root, PYTHONPATH=.;code + ARMY core on path):
  from lineages...kag_teachers.full_obs_reason_hook import collect_full_obs_reason_labels
  X, y_act, aux_list, meta = collect_full_obs_reason_labels(days, max_days=10)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LINEAGE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(_LINEAGE))
_ARMY_CORE = r"C:\Users\user\OneDrive\Desktop\ARMY\01_SYSTEM\packages\core"
for _p in (_ROOT, os.path.join(_ROOT, "code"), _ARMY_CORE):
    if _p and _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM

CKPT = os.path.join(_LINEAGE, "checkpoints")
OUT_DIR = os.path.join(CKPT, "full_obs_reason")
LABELS_PATH = os.path.join(OUT_DIR, "FULL_OBS_REASON_LABELS__lineage.jsonl")
REPORT_PATH = os.path.join(OUT_DIR, "FULL_OBS_REASON_COLLECT__latest.json")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _army_process(obs: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
    from markos_core.kag_mark.full_obs_pipeline import process_full_obs_bar

    return process_full_obs_bar(obs.tolist(), **kwargs)


def collect_full_obs_reason_labels(
    days: Sequence[Tuple[str, Any]],
    *,
    target: float = 2.0,
    risk: float = 3.0,
    decide_every: int = 25,
    max_days: int = 10,
    max_bars_per_day: int = 40,
    seed: int = 42,
    write: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Walk days with full_obs=True; emit reason+pattern+aux labels every decision bar.

    Returns
    -------
    X : (N, 168) float32 — full observation the bot MUST see
    y_act : (N,) int64 — HOLD/BUY/SELL from reason (or soul teacher if provided)
    aux_list : multi-head targets per row (topology, wait, roles, motifs, …)
    meta : collection report
    """
    xs: List[np.ndarray] = []
    ys: List[int] = []
    aux_list: List[Dict[str, Any]] = []
    n_reason_ok = 0
    n_l2l_ok = 0
    n_fail = 0

    os.makedirs(OUT_DIR, exist_ok=True)
    if write and os.path.isfile(LABELS_PATH):
        # append mode collection — do not wipe
        pass

    for i, (date_str, m1) in enumerate(list(days)[:max_days]):
        day = GoalEquityDay(
            m1,
            target_pct=target,
            risk_pct=risk,
            date_str=str(date_str),
            decide_every=decide_every,
            eyes_mode="mark_doctrine",
            mark_clone=False,
            mark_soul=True,
            full_obs=True,
        )
        names = list(getattr(day, "_sig_names", None) or [])
        bars = 0
        for t_bar in day.runner.decision_indices():
            if day.banked or day.dead or bars >= max_bars_per_day:
                break
            obs = np.asarray(day.observe(t_bar), dtype=np.float32).reshape(-1)
            if obs.size < MARK_FULL_DIM:
                pad = np.zeros(MARK_FULL_DIM, dtype=np.float32)
                pad[: obs.size] = obs
                obs = pad
            else:
                obs = obs[:MARK_FULL_DIM]

            teacher = int(day.recommended_action(t_bar))
            try:
                day._ensure_signal_panel()
                names = list(getattr(day, "_sig_names", None) or names)
            except Exception:
                pass

            try:
                out = _army_process(
                    obs,
                    agent_names=names or None,
                    day=str(date_str),
                    bar_index=int(t_bar),
                    teacher_action=teacher,
                    persist=write,
                    write_label_row=False,  # we write lineage path below
                )
                aux = dict(out.get("aux_targets") or {})
                # Prefer Teacher 1 soul action for BC act head when available;
                # keep reason topology/wait for multi-head (learn≠copy)
                y = teacher
                aux["teacher1_action"] = teacher
                aux["reason_policy_action"] = aux.get("policy_action")
                aux["reason_trace_id"] = (out.get("reason") or {}).get("trace_id")
                aux["motifs"] = (out.get("pattern_graph") or {}).get("motifs")
                aux["hop_depth"] = (out.get("pattern_graph") or {}).get("hop_depth")
                aux["learn_to_learn_ok"] = (out.get("learn_to_learn") or {}).get(
                    "learn_to_learn_ok"
                )
                if (out.get("learn_to_learn") or {}).get("learn_to_learn_ok"):
                    n_l2l_ok += 1
                n_reason_ok += 1
            except Exception as e:
                n_fail += 1
                y = teacher
                aux = {
                    "act": "hold_fallback",
                    "topology": "no_trade",
                    "wait_subtype": "no_trade",
                    "policy_action": y,
                    "error": str(e),
                    "sees_all_obs": True,
                }

            xs.append(obs.astype(np.float32))
            ys.append(int(y))
            aux_list.append(aux)

            if write:
                with open(LABELS_PATH, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "day": str(date_str),
                                "bar": int(t_bar),
                                "obs": obs.tolist(),
                                "y_act": int(y),
                                "aux": aux,
                            },
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )

            day.step_action(t_bar, teacher)
            bars += 1

    if not xs:
        X = np.zeros((0, MARK_FULL_DIM), dtype=np.float32)
        y_arr = np.zeros((0,), dtype=np.int64)
    else:
        X = np.stack(xs, axis=0)
        y_arr = np.asarray(ys, dtype=np.int64)

    meta = {
        "ts": _utcnow(),
        "n_rows": int(X.shape[0]),
        "obs_dim": MARK_FULL_DIM,
        "n_days": min(max_days, len(days)),
        "n_reason_ok": n_reason_ok,
        "n_l2l_ok": n_l2l_ok,
        "n_fail": n_fail,
        "labels_path": LABELS_PATH,
        "law": "full_obs always; multi-head labels; agents=sensors; PROVEN never touched",
        "success": X.shape[0] > 0 and X.shape[1] == MARK_FULL_DIM,
    }
    if write:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
    return X, y_arr, aux_list, meta


def try_collect_practice(
    *,
    max_days: int = 5,
    max_bars_per_day: int = 20,
) -> Dict[str, Any]:
    """Convenience: load practice days if data present; else return dry status."""
    try:
        from lineages.adaptive_rl_brain_7_31_26.equity_day import (
            load_calendar_days,
            split_practice_forward,
        )

        days = load_calendar_days()
        practice, _forward = split_practice_forward(days)
        if not practice:
            return {"ok": False, "error": "no practice days"}
        X, y, aux, meta = collect_full_obs_reason_labels(
            practice,
            max_days=max_days,
            max_bars_per_day=max_bars_per_day,
            write=True,
        )
        meta["X_shape"] = list(X.shape)
        meta["y_hist"] = {
            "hold": int((y == 0).sum()) if y.size else 0,
            "buy": int((y == 1).sum()) if y.size else 0,
            "sell": int((y == 2).sum()) if y.size else 0,
        }
        meta["aux_with_topology"] = sum(1 for a in aux if a.get("topology"))
        return {"ok": True, **meta}
    except Exception as e:
        return {"ok": False, "error": str(e)}
