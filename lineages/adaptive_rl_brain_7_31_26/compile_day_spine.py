"""S0 — Compile Mark soul plans into sparse Day Spines (Spine Shadow).

Pure spine objects: day fields + sparse events. No bar I/O, no training.
Round-trip: soul_plan dict → DaySpine → action plan must preserve fire/add bars.

Doctrine: 01_SYSTEM/Fable 5 Alternate — Spine Shadow.md
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
SPINE_DIR = os.path.join(_HERE, "checkpoints", "spines")
SPINE_INDEX = os.path.join(SPINE_DIR, "SPINE_INDEX__latest.json")

SIDE_NAME = {ACTION_HOLD: "HOLD", ACTION_BUY: "BUY", ACTION_SELL: "SELL"}
SIDE_FROM_NAME = {"HOLD": ACTION_HOLD, "BUY": ACTION_BUY, "SELL": ACTION_SELL, "+1": ACTION_BUY, "-1": ACTION_SELL}

# Map soul size dials → discrete size_bucket labels (spine product)
SIZE_BUCKETS: List[Tuple[str, float, float]] = [
    ("micro", 0.25, 0.20),
    ("base", 0.35, 0.25),
    ("std", 0.50, 0.35),
    ("lag_add", 0.65, 0.45),
    ("heavy", 0.80, 0.55),
    ("max", 1.00, 0.70),
]


def size_bucket_for(ruf: float, cap: float) -> str:
    """Nearest SIZE_GRID dial → bucket name."""
    best = "base"
    best_d = 1e9
    for name, gr, gc in SIZE_BUCKETS:
        d = abs(float(ruf) - gr) + abs(float(cap) - gc)
        if d < best_d:
            best_d = d
            best = name
    return best


def dials_for_bucket(bucket: str) -> Tuple[float, float]:
    for name, gr, gc in SIZE_BUCKETS:
        if name == bucket:
            return gr, gc
    return 0.35, 0.25


@dataclass
class SpineEvent:
    t: int  # decision bar index
    kind: str  # wait_loaded | fire | add | bank | kill | hold_on_spine
    side: Optional[str] = None  # BUY|SELL|+1|-1
    size_bucket: Optional[str] = None
    tide: str = "0"
    topo: str = "unknown"
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # drop nulls for compact JSON
        return {k: v for k, v in d.items() if v is not None and v != ""}


@dataclass
class DaySpine:
    day: str
    target_pct: float
    risk_pct: float
    tide_day: str = "0"
    regime_day: str = "unknown"
    events: List[SpineEvent] = field(default_factory=list)
    clue_prior: Dict[str, float] = field(default_factory=lambda: {"force_family": 1.0, "noise_agents": 0.1})
    mark_source: str = "soul_plan"
    risk_use_frac: float = 0.35
    per_trade_cap_pct: float = 0.25
    mode: str = "single"
    cleared: Optional[bool] = None
    breached: Optional[bool] = None
    side: Optional[str] = None
    t1: Optional[int] = None
    t2: Optional[int] = None
    # Full dense plan for lossless re-execution (oracle path)
    plan: Optional[Dict[int, int]] = None
    decision_indices: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "day": self.day,
            "target_pct": float(self.target_pct),
            "risk_pct": float(self.risk_pct),
            "tide_day": self.tide_day,
            "regime_day": self.regime_day,
            "events": [e.to_dict() if isinstance(e, SpineEvent) else e for e in self.events],
            "clue_prior": dict(self.clue_prior),
            "mark_source": self.mark_source,
            "risk_use_frac": self.risk_use_frac,
            "per_trade_cap_pct": self.per_trade_cap_pct,
            "mode": self.mode,
            "side": self.side,
            "t1": self.t1,
            "t2": self.t2,
            "cleared": self.cleared,
            "breached": self.breached,
        }
        if self.plan is not None:
            d["plan"] = {str(k): int(v) for k, v in self.plan.items()}
        if self.decision_indices is not None:
            d["decision_indices"] = list(self.decision_indices)
        return d

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "DaySpine":
        events = []
        for e in raw.get("events") or []:
            events.append(
                SpineEvent(
                    t=int(e["t"]),
                    kind=str(e["kind"]),
                    side=e.get("side"),
                    size_bucket=e.get("size_bucket"),
                    tide=str(e.get("tide", "0")),
                    topo=str(e.get("topo", "unknown")),
                    reason=str(e.get("reason", "")),
                )
            )
        plan = raw.get("plan")
        plan_i = None
        if plan is not None:
            plan_i = {int(k): int(v) for k, v in plan.items()}
        return cls(
            day=str(raw["day"]),
            target_pct=float(raw["target_pct"]),
            risk_pct=float(raw["risk_pct"]),
            tide_day=str(raw.get("tide_day", "0")),
            regime_day=str(raw.get("regime_day", "unknown")),
            events=events,
            clue_prior=dict(raw.get("clue_prior") or {"force_family": 1.0, "noise_agents": 0.1}),
            mark_source=str(raw.get("mark_source", "soul_plan")),
            risk_use_frac=float(raw.get("risk_use_frac") or 0.35)
            if raw.get("risk_use_frac") not in (None, "dynamic")
            else 0.35,
            per_trade_cap_pct=float(raw.get("per_trade_cap_pct") or 0.25)
            if raw.get("per_trade_cap_pct") not in (None, "dynamic")
            else 0.25,
            mode=str(raw.get("mode", "single")),
            cleared=raw.get("cleared"),
            breached=raw.get("breached"),
            side=raw.get("side"),
            t1=int(raw["t1"]) if raw.get("t1") is not None else None,
            t2=int(raw["t2"]) if raw.get("t2") is not None else None,
            plan=plan_i,
            decision_indices=[int(x) for x in raw["decision_indices"]]
            if raw.get("decision_indices")
            else None,
        )


def compile_spine_from_soul(
    date: str,
    target_pct: float,
    risk_pct: float,
    mark: Dict[str, Any],
    *,
    decision_indices: Optional[Sequence[int]] = None,
) -> DaySpine:
    """Compile a DaySpine from execute_mark_soul_day / oracle-cache blob.

    Events are sparse: wait_loaded before first fire, fire at t1, add at t2,
    hold_on_spine at other decision bars, bank if mark banked.
    """
    source = str(mark.get("source") or "soul_plan")
    mode = str(mark.get("mode") or "single")
    side = mark.get("side")
    if side is None and mark.get("t1") is not None and mark.get("plan"):
        plan0 = {int(k): int(v) for k, v in mark["plan"].items()}
        a = int(plan0.get(int(mark["t1"]), ACTION_HOLD))
        side = SIDE_NAME.get(a, "HOLD")

    ruf = mark.get("risk_use_frac")
    cap = mark.get("per_trade_cap_pct")
    if ruf in (None, "dynamic"):
        ruf_f, cap_f = 0.35, 0.25
    else:
        ruf_f, cap_f = float(ruf), float(cap if cap not in (None, "dynamic") else 0.25)
    bucket = size_bucket_for(ruf_f, cap_f)

    plan: Optional[Dict[int, int]] = None
    if mark.get("plan") is not None:
        plan = {int(k): int(v) for k, v in mark["plan"].items()}

    indices: List[int]
    if decision_indices is not None:
        indices = [int(x) for x in decision_indices]
    elif plan is not None:
        indices = sorted(plan.keys())
    else:
        indices = []

    t1 = int(mark["t1"]) if mark.get("t1") is not None else None
    t2 = int(mark["t2"]) if mark.get("t2") is not None else None

    events: List[SpineEvent] = []
    if t1 is not None and indices:
        # wait_loaded on decision bars strictly before first fire
        for tb in indices:
            if tb >= t1:
                break
            events.append(
                SpineEvent(
                    t=int(tb),
                    kind="wait_loaded",
                    side=None,
                    size_bucket=None,
                    tide="+1" if side == "BUY" else ("-1" if side == "SELL" else "0"),
                    topo="slingshot_load",
                    reason="pre_fire_hold",
                )
            )
        # first fire
        events.append(
            SpineEvent(
                t=int(t1),
                kind="fire",
                side=str(side) if side else None,
                size_bucket=bucket,
                tide="+1" if side == "BUY" else ("-1" if side == "SELL" else "0"),
                topo="slingshot_release",
                reason="soul_entry",
            )
        )
        if t2 is not None and t2 != t1:
            events.append(
                SpineEvent(
                    t=int(t2),
                    kind="add",
                    side=str(side) if side else None,
                    size_bucket=bucket,
                    tide="+1" if side == "BUY" else ("-1" if side == "SELL" else "0"),
                    topo="force_add",
                    reason="soul_add",
                )
            )
            # hold_on_spine between fire and add / after add for remaining bars with plan HOLD
            for tb in indices:
                if tb == t1 or tb == t2 or tb < t1:
                    continue
                if plan is not None and int(plan.get(tb, ACTION_HOLD)) != ACTION_HOLD:
                    # extra directional bar not captured as t1/t2 — treat as add
                    events.append(
                        SpineEvent(
                            t=int(tb),
                            kind="add",
                            side=SIDE_NAME.get(int(plan[tb]), str(side)),
                            size_bucket=bucket,
                            topo="extra_dir",
                            reason="plan_extra_dir",
                        )
                    )
                else:
                    events.append(
                        SpineEvent(
                            t=int(tb),
                            kind="hold_on_spine",
                            topo="in_trade_or_done",
                            reason="spine_hold",
                        )
                    )
        else:
            for tb in indices:
                if tb <= t1:
                    continue
                events.append(
                    SpineEvent(
                        t=int(tb),
                        kind="hold_on_spine",
                        topo="in_trade_or_done",
                        reason="spine_hold",
                    )
                )
        if mark.get("banked"):
            events.append(
                SpineEvent(
                    t=int(indices[-1]) if indices else int(t1),
                    kind="bank",
                    reason="goal",
                    topo="done_bank",
                )
            )
    elif source == "soul_online_fallback":
        # No sparse plan — single wait marker; oracle uses online path
        events.append(
            SpineEvent(
                t=0,
                kind="wait_loaded",
                topo="online_fallback",
                reason="no_sparse_plan",
            )
        )
    else:
        events.append(
            SpineEvent(t=0, kind="wait_loaded", topo="empty", reason="no_events")
        )

    tide_day = "+1" if side == "BUY" else ("-1" if side == "SELL" else "0")
    return DaySpine(
        day=str(date),
        target_pct=float(target_pct),
        risk_pct=float(risk_pct),
        tide_day=tide_day,
        regime_day="soul_compiled",
        events=events,
        clue_prior={"force_family": 1.0, "noise_agents": 0.1},
        mark_source=source,
        risk_use_frac=ruf_f,
        per_trade_cap_pct=cap_f,
        mode=mode,
        cleared=bool(mark["cleared"]) if mark.get("cleared") is not None else None,
        breached=bool(mark["breached"]) if mark.get("breached") is not None else None,
        side=str(side) if side else None,
        t1=t1,
        t2=t2,
        plan=plan,
        decision_indices=indices or None,
    )


def spine_to_plan(spine: DaySpine) -> Dict[int, int]:
    """Reconstruct dense action plan from spine (lossless if plan embedded)."""
    if spine.plan is not None:
        return {int(k): int(v) for k, v in spine.plan.items()}
    plan: Dict[int, int] = {}
    if spine.decision_indices:
        for tb in spine.decision_indices:
            plan[int(tb)] = ACTION_HOLD
    for ev in spine.events:
        if ev.kind in ("fire", "add"):
            side = ev.side or spine.side or "HOLD"
            if side in ("BUY", "+1"):
                plan[int(ev.t)] = ACTION_BUY
            elif side in ("SELL", "-1"):
                plan[int(ev.t)] = ACTION_SELL
            else:
                plan[int(ev.t)] = ACTION_HOLD
        elif ev.kind in ("wait_loaded", "hold_on_spine", "bank", "kill"):
            plan.setdefault(int(ev.t), ACTION_HOLD)
    return plan


def fire_times(spine: DaySpine) -> List[int]:
    return [int(e.t) for e in spine.events if e.kind in ("fire", "add")]


def wait_times(spine: DaySpine) -> List[int]:
    return [int(e.t) for e in spine.events if e.kind == "wait_loaded"]


def classify_spine_error(
    *,
    spine: DaySpine,
    policy_fire_ts: Sequence[int],
    policy_n_entries: int,
    policy_award: bool,
    policy_breached: bool,
) -> str:
    """Map autopsy-style miss to spine error class (for error cards)."""
    if policy_breached:
        return "breach_thrash"
    if policy_award:
        return "aligned"
    fires = fire_times(spine)
    if not fires:
        return "no_spine_fire"
    if policy_n_entries == 0:
        return "false_hold"
    # thrash / extra fires before timing subclass
    if policy_n_entries > len(fires) + 1:
        return "false_fire"
    p_first = min(policy_fire_ts) if policy_fire_ts else None
    s_first = min(fires)
    if p_first is not None and p_first > s_first + 50:
        return "late_entry"
    if p_first is not None and p_first < s_first - 25:
        return "early_entry"
    # size / timing residual
    return "wrong_size_or_timing"


def write_spine(spine: DaySpine, out_dir: str = SPINE_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    key = f"{spine.day}__t{spine.target_pct}_r{spine.risk_pct}".replace(".", "p")
    path = os.path.join(out_dir, f"SPINE__{key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spine.to_dict(), f, indent=2)
    return path


def write_spine_index(
    spines: List[DaySpine],
    paths: List[str],
    *,
    out_path: str = SPINE_INDEX,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    blob = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "n_spines": len(spines),
        "meta": meta or {},
        "items": [
            {
                "day": s.day,
                "target_pct": s.target_pct,
                "risk_pct": s.risk_pct,
                "path": paths[i],
                "n_events": len(s.events),
                "mark_source": s.mark_source,
                "side": s.side,
                "t1": s.t1,
                "t2": s.t2,
                "mode": s.mode,
                "cleared": s.cleared,
            }
            for i, s in enumerate(spines)
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)
    return out_path


def load_spine(path: str) -> DaySpine:
    with open(path, encoding="utf-8") as f:
        return DaySpine.from_dict(json.load(f))


def load_spine_index(path: str = SPINE_INDEX) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
