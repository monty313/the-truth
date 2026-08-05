"""Mark sense gate — policy shares Mark's capital/force judgment.

HITL truth:
  - Mark HOLD when force not ready → do not invent early opens
  - Never open against HTF force
  - Never open when half to the floor (danger)
  - Do NOT slave every bar to online thrash recommended_action
    (soul plans are sparse; online teacher is noisier)

Soul = policy means: same *laws*, learned weights, force/capital gates.
"""
from __future__ import annotations

from typing import Any, Optional

from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
)


def _is_dead_regime(regime: str) -> bool:
    """True only for regimes that mean sit out — not 'flat_undefined' false positive.

    BUGFIX 2026-08-05: bare ``\"flat\" in reg`` blocked ``flat_undefined`` and killed
    Mark-agreed SELLs (e.g. 2026-04-06 t=745 pol=S rec=S → gated HOLD). That
    broke award streaks: policy had the right action; gate erased it.
    """
    reg = (regime or "").lower().strip()
    if not reg:
        return False
    # exact / token dead regimes
    dead = {
        "chop",
        "flat",
        "range",
        "choppy",
        "no_trade",
        "dead",
    }
    if reg in dead:
        return True
    if reg.startswith("chop") or reg.endswith("_chop"):
        return True
    # explicit multi-token
    if "chop" in reg and "undefined" not in reg:
        return True
    return False


def mark_force_gate_action(
    policy_action: int,
    *,
    side: Optional[int],
    equity_pct: float,
    risk_pct: float,
    force_dir: float = 0.0,
    m_conf: float = 1.0,
    regime: str = "",
    recommended: int = ACTION_HOLD,
) -> int:
    """Filter policy action by Mark force + capital sense.

    When online Mark ``recommended`` agrees with policy side, only the capital
    danger wall may block — force/regime noise must not erase Mark-agreed entries
    (soul-plan wins were being zeroed by flat_undefined + force=0).
    """
    pol = int(policy_action)
    rec = int(recommended)
    danger = max(0.0, -float(equity_pct)) / max(float(risk_pct), 1e-6)
    reg = (regime or "").lower()

    if side is None and pol in (ACTION_BUY, ACTION_SELL):
        if danger >= 0.45:
            return ACTION_HOLD
        # Mark online agrees → allow (capital already checked)
        if rec == pol:
            return pol
        if _is_dead_regime(reg):
            return ACTION_HOLD
        if float(m_conf) <= 1e-9:
            return ACTION_HOLD
        # against clear HTF force
        if force_dir > 0.5 and pol == ACTION_SELL:
            return ACTION_HOLD
        if force_dir < -0.5 and pol == ACTION_BUY:
            return ACTION_HOLD
        return pol

    if side is not None and pol in (ACTION_BUY, ACTION_SELL):
        want = 1 if pol == ACTION_BUY else -1
        if want != int(side):
            # reverse: only if force agrees / not dead
            if rec == pol and danger < 0.55:
                return pol
            if force_dir > 0.5 and want < 0:
                return ACTION_HOLD
            if force_dir < -0.5 and want > 0:
                return ACTION_HOLD
            if danger >= 0.55:
                return ACTION_HOLD
            if _is_dead_regime(reg):
                return ACTION_HOLD
        return pol
    return pol


def mark_aligned_action(
    policy_action: int,
    mark_action: int,
    *,
    side: Optional[int],
    equity_pct: float = 0.0,
    target_pct: float = 2.0,
    risk_pct: float = 3.0,
    force_dir: float = 0.0,
    m_conf: float = 1.0,
    regime: str = "",
    strict_teacher: bool = False,
) -> int:
    """Default: force/capital gate. Optional strict_teacher for HITL replay only."""
    if strict_teacher:
        # Legacy strict: Mark HOLD blocks open (use for BC label gen, not long run)
        pol = int(policy_action)
        mark = int(mark_action)
        if side is None:
            if mark == ACTION_HOLD:
                return ACTION_HOLD
            if mark in (ACTION_BUY, ACTION_SELL) and pol == ACTION_HOLD:
                return mark
            if mark in (ACTION_BUY, ACTION_SELL) and pol != mark:
                return mark
        return mark_force_gate_action(
            pol,
            side=side,
            equity_pct=equity_pct,
            risk_pct=risk_pct,
            force_dir=force_dir,
            m_conf=m_conf,
            regime=regime,
            recommended=mark,
        )
    return mark_force_gate_action(
        int(policy_action),
        side=side,
        equity_pct=equity_pct,
        risk_pct=risk_pct,
        force_dir=force_dir,
        m_conf=m_conf,
        regime=regime,
        recommended=int(mark_action),
    )


def doctrine_fields(day: Any, t: int) -> tuple[float, float, str]:
    """force_dir in {-1,0,1}, m_conf, regime str from last doctrine."""
    try:
        day._raw_structure_sig(t)
    except Exception:
        pass
    dec = getattr(getattr(day, "runner", None), "last_doctrine", None)
    if dec is None:
        return 0.0, 1.0, ""
    fd = getattr(dec, "force_dir", None)
    name = str(getattr(fd, "value", fd) or "").lower()
    if "bull" in name:
        f = 1.0
    elif "bear" in name:
        f = -1.0
    else:
        f = 0.0
    m = float(getattr(dec, "m_conf", 1.0) or 0.0)
    reg = str(getattr(getattr(dec, "regime", None), "value", getattr(dec, "regime", "")) or "")
    return f, m, reg


def policy_mark_step(
    day: Any,
    t: int,
    policy: Any,
    *,
    aligned: bool = True,
    strict_teacher: bool = False,
) -> int:
    obs = day.observe(t)
    act, _ = policy.act(obs, greedy=True)
    act = int(act)
    if not aligned:
        return act
    mark = int(day.recommended_action(t))
    f, m, reg = doctrine_fields(day, t)
    eq = float(day.equity_pct(float(day._close[t])))
    return mark_aligned_action(
        act,
        mark,
        side=day.side,
        equity_pct=eq,
        target_pct=float(day.target),
        risk_pct=float(day.risk),
        force_dir=f,
        m_conf=m,
        regime=reg,
        strict_teacher=strict_teacher,
    )
