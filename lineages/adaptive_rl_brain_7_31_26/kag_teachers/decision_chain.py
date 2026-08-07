"""Decision chain runtime — teachers + student target (the-truth).

Order fixed: tide → regime → breath/launch → act → finish.
Pins OFFICIAL_SETS via perception.sets (MARK SETS LAW).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from lineages.adaptive_rl_brain_7_31_26.perception.sets import OFFICIAL_SETS

CHAIN = ("tide", "regime", "breath_launch", "act", "finish")


def pack_official_sets(obs: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    obs = dict(obs or {})
    per_set = obs.get("sets") or {}
    packs = []
    for s in OFFICIAL_SETS:
        raw = per_set.get(str(s.set_id)) or per_set.get(s.set_id) or {}
        packs.append(
            {
                "set_id": s.set_id,
                "anchor": s.entry_tf,
                "support": list(s.confirmation_tfs),
                "stack": list(s.tfs),
                "force_side": int(raw.get("force_side", 0)),
                "inertia_with": bool(raw.get("inertia_with", False)),
                "velocity_against": bool(raw.get("velocity_against", False)),
                "velocity_with": bool(raw.get("velocity_with", False)),
                "g_fixed": bool(raw.get("g_fixed", True)),
                "g_flip": bool(raw.get("g_flip", False)),
                "efficiency_ok": bool(raw.get("efficiency_ok", True)),
                "regime": str(raw.get("regime", "undefined")),
                "sensors": list(raw.get("sensors", [])),
            }
        )
    return packs


def _topology(p: Mapping[str, Any]) -> str:
    if p.get("g_flip") or (not p.get("g_fixed") and int(p.get("force_side", 0)) == 0):
        return "collapse"
    if str(p.get("regime", "")).lower() in ("chop", "range", "undefined") or not p.get(
        "efficiency_ok", True
    ):
        return "chop"
    if p.get("inertia_with") and p.get("velocity_against") and p.get("g_fixed"):
        return "slingshot_load"
    if p.get("inertia_with") and p.get("velocity_with") and p.get("g_fixed"):
        return "slingshot_release"
    if p.get("velocity_with") and p.get("g_fixed") and int(p.get("force_side", 0)) != 0:
        return "slingshot_release"
    return "no_trade"


def _act(topology: str, tide: str) -> str:
    if topology == "slingshot_load":
        return "wait_loaded"
    if topology in ("slingshot_release", "launch"):
        if tide == "long_only":
            return "fire_buy"
        if tide == "short_only":
            return "fire_sell"
        return "wait_no_trade"
    if topology == "collapse":
        return "kill"
    if topology == "chop":
        return "wait_no_trade"
    return "wait_no_trade"


def decision_chain(
    packs: Sequence[Mapping[str, Any]] | None = None,
    *,
    obs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if packs is None:
        packs = pack_official_sets(obs)
    results = []
    for p in packs:
        side = int(p.get("force_side", 0))
        tide = "long_only" if side > 0 else ("short_only" if side < 0 else "flat")
        topo = _topology(p)
        act = _act(topo, tide)
        if act == "fire_buy" and tide != "long_only":
            act = "wait_no_trade"
        if act == "fire_sell" and tide != "short_only":
            act = "wait_no_trade"
        wait_subtype = (
            "loaded"
            if act == "wait_loaded"
            else ("no_trade" if act == "wait_no_trade" else None)
        )
        results.append(
            {
                "set_id": p.get("set_id"),
                "tide": tide,
                "topology": topo,
                "act": act,
                "wait_subtype": wait_subtype,
            }
        )
    acts = [r["act"] for r in results]
    if "wait_loaded" in acts and not any(a.startswith("fire_") for a in acts):
        global_act = "wait_loaded"
    elif acts.count("fire_buy") >= acts.count("fire_sell") and "fire_buy" in acts:
        global_act = "fire_buy"
    elif "fire_sell" in acts:
        global_act = "fire_sell"
    elif "kill" in acts:
        global_act = "kill"
    else:
        global_act = "wait_no_trade"
    return {
        "chain_order": list(CHAIN),
        "sets": results,
        "global_act": global_act,
    }


def bread_and_butter_obs(*, set_id: int = 2, side: int = 1) -> dict[str, Any]:
    return {
        "sets": {
            set_id: {
                "force_side": side,
                "inertia_with": True,
                "velocity_against": True,
                "velocity_with": False,
                "g_fixed": True,
                "efficiency_ok": True,
                "regime": "bull_trend" if side > 0 else "bear_trend",
                "sensors": [
                    {"name": "CCI", "period": 100, "tf": "1h", "role": "inertia", "novel": False},
                    {"name": "CCI", "period": 30, "tf": "5m", "role": "velocity", "novel": False},
                    {"name": "SMA", "period": 50, "tf": "1h", "role": "force", "novel": False},
                ],
            }
        }
    }
