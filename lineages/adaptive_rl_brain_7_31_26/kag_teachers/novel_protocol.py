"""Novel indicator protocol (the-truth) — pure assign_role.

Mirrors ARMY markos_core.kag_mark.novel_protocol; uses local sets law.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from lineages.adaptive_rl_brain_7_31_26.perception.sets import OFFICIAL_SETS

_ANCHOR_TFS = {s.entry_tf for s in OFFICIAL_SETS}
_SUPPORT_TFS = {tf for s in OFFICIAL_SETS for tf in s.confirmation_tfs}

_OSC_HINTS = ("rsi", "cci", "stoch", "wpr", "willr", "momentum", "demarker", "dem", "macd", "osc")
_MID_HINTS = ("sma", "ema", "ma", "lwma", "hull", "mid")
_BAND_HINTS = ("bb", "bollinger", "envelope", "donchian", "keltner", "band")
_VOL_HINTS = ("atr", "stdev", "std", "width", "hv", "vol_")
_DIR_HINTS = ("adx", "+di", "-di", "di_", "dmi")
_VOLUME_HINTS = ("volume", "obv", "mfi", "flow")

ROLES = frozenset(
    {"force", "inertia", "velocity", "equilibrium", "regime_gate", "expansion", "volume_confirm"}
)


@dataclass(frozen=True)
class RoleAssignment:
    role: str
    confidence: str
    mask_tide: bool
    why_role: str
    novel: bool = True
    known_force_wins: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _shape_role_bias(name: str) -> tuple[str | None, str]:
    n = name.lower()
    if any(h in n for h in _VOLUME_HINTS):
        return "volume_confirm", "volume-like"
    if any(h in n for h in _DIR_HINTS):
        return "regime_gate", "ADX-like"
    if any(h in n for h in _VOL_HINTS):
        return "regime_gate", "volatility"
    if any(h in n for h in _BAND_HINTS):
        return "expansion", "band/envelope"
    if any(h in n for h in _MID_HINTS):
        return "equilibrium", "mid/MA"
    if any(h in n for h in _OSC_HINTS):
        return "velocity", "oscillator"
    return None, "unknown shape"


def assign_role(
    sensor: Mapping[str, Any],
    set_context: Mapping[str, Any] | None = None,
    known_roles: Sequence[Mapping[str, Any]] | None = None,
) -> RoleAssignment:
    name = str(sensor.get("name") or "UNKNOWN")
    tf = str(sensor.get("tf") or sensor.get("timeframe") or "").lower()
    period = sensor.get("period")
    known_roles = list(known_roles or [])
    shape_role, shape_why = _shape_role_bias(name)

    slot = None
    if tf in _ANCHOR_TFS:
        slot = "anchor"
    elif tf in _SUPPORT_TFS:
        slot = "support"

    role = "velocity"
    conf = "medium"
    reasons: list[str] = []

    if slot is None and tf:
        return RoleAssignment(
            role="velocity",
            confidence="low",
            mask_tide=True,
            why_role="TF outside official sets — mask",
        )

    # period relativity vs peers
    period_role = None
    if period is not None:
        peers = [float(p["period"]) for p in known_roles if p.get("period") is not None]
        if peers:
            if float(period) <= min(peers):
                period_role = "velocity"
            elif float(period) >= max(peers):
                period_role = "inertia"

    if slot == "support":
        if shape_role in ("regime_gate", "expansion", "volume_confirm"):
            role = shape_role
            reasons.append(shape_why)
        elif period_role == "inertia":
            role = "inertia"
            reasons.append("slow period HTF")
        elif shape_role == "equilibrium":
            role = "force"
            reasons.append("HTF mid → force")
        else:
            role = "force"
            reasons.append("support TF force bias")
        conf = "high" if (shape_role or period_role) else "medium"
    else:
        if period_role == "velocity" or shape_role in (None, "velocity"):
            role = "velocity"
            reasons.append(shape_why if shape_role else "LTF anchor velocity bias")
        elif shape_role:
            role = shape_role if shape_role != "equilibrium" else "velocity"
            reasons.append(shape_why)
        conf = "high" if (shape_role or period_role) else "medium"

    if shape_role is None and period_role is None:
        conf = "low"

    known_force = any(
        str(r.get("role", "")).lower() in ("force", "inertia") and not r.get("novel", False)
        for r in known_roles
    )
    mask_tide = conf != "high" or role in ("velocity", "volume_confirm") or not known_force
    if conf == "high" and role in ("force", "inertia") and not known_force:
        mask_tide = True
        conf = "medium"
        reasons.append("novel never defines tide alone")

    if role not in ROLES:
        role = "velocity"
        conf = "low"
        mask_tide = True

    return RoleAssignment(
        role=role,
        confidence=conf,
        mask_tide=mask_tide,
        why_role="; ".join(reasons) or "default",
        novel=bool(sensor.get("novel", True)),
        known_force_wins=True,
    )
