"""Learn-to-learn for path issues — stop re-learning the same failure every day.

Problem we keep hitting:
  Teach one MWT day → day converts → pack dies (35→32).
  Root: day *answers* without path *principles* (wait_loaded / fire_window / anti_thrash).

L2L here:
  1. Classify every disagree into a PATH CLASS (transferable), not a calendar memo.
  2. Train act + path_class heads together (shared trunk).
  3. learn≠copy gate: high act match + low class match → COPYING → REJECT.
  4. Pattern memory: after each cycle, boost the class that still fails next round
     so we do not re-discover "early fire" from scratch every focus day.

pt5: learn principles ≠ copy answers. Agents=sensors. PROVEN untouched.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay, load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.fable_50d_mark_match_loop import load_policy, save_policy
from lineages.adaptive_rl_brain_7_31_26.fable_50d_rapid import (
    award_self,
    get_plan,
    load_oracle,
    score_policy,
)
from lineages.adaptive_rl_brain_7_31_26.kag_teachers.student_interface import check_learn_not_copy
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.perception.sets import (
    MARK_SETS_LAW,
    assert_mark_sets_law,
    mark_sets_law_table,
)
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_HOLD,
    ACTION_BUY,
    ACTION_SELL,
    Channel1Policy,
    N_ACTIONS,
)
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import match_rate

# Immutable timeframe stacks (MARK SETS LAW) — LTF first | HTF last two
#   Set1 micro:    1m  | 15m, 30m
#   Set2 intraday: 5m  | 30m, 1h
#   Set3 swing:    15m | 1h,  4h
#   Set4 macro:    30m | 4h,  1d
# LTF job = pullback / continuation / add; HTF job = strong bull|bear trend confirm.
# Scan ALL four under mark_doctrine (never set-2-only for Mark path).

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")
BEST = os.path.join(OUT, "BEST__latest.json")
L2L_MEM = os.path.join(OUT, "L2L_PATH_MEMORY.jsonl")
L2L_STATE = os.path.join(OUT, "L2L_PATH_STATE__latest.json")
L2L_REPORT = os.path.join(OUT, "L2L_PATH_CYCLE__latest.md")
L2L_CKPT = os.path.join(_HERE, "checkpoints", "mark_shadow_l2l_v1.pt")

# Learn-to-learn core skill (corrected):
#   While HTF is strong BULL or BEAR, identify on LTF:
#     • pullback  (LTF opposes HTF) → wait / HOLD
#     • continuation (LTF aligns HTF) → fire with HTF
# HTF not strong → do not invent (htf_not_strong).
PATH_CLASSES = [
    "ltf_pullback_htf_strong",  # HTF clear + LTF opposite → pullback: wait is skill
    "ltf_continuation_htf_strong",  # HTF clear + LTF same → continuation: fire with tide
    "htf_not_strong",  # no strong HTF bull/bear → no invent
    "anti_thrash",  # policy fired when should wait (esp pullback / weak HTF)
    "miss_continuation",  # policy HOLD when should continue with HTF
    "hold_on_spine",  # award protect / banked path agree HOLD
]
CLASS_TO_I = {c: i for i, c in enumerate(PATH_CLASSES)}
N_PATH = len(PATH_CLASSES)

# Backward-compatible aliases if old memory rows mention prior names
_CLASS_ALIASES = {
    "wait_loaded": "ltf_pullback_htf_strong",
    "fire_window": "ltf_continuation_htf_strong",
    "pullback_hold": "ltf_pullback_htf_strong",
    "continuation_fire": "ltf_continuation_htf_strong",
    "miss_fire": "miss_continuation",
}


@dataclass
class PathLesson:
    """One bar lesson: act + path class (principle), not day id."""

    path_class: str
    mark_act: int
    policy_act: int
    weight: float
    principle_ids: Tuple[str, ...]
    htf_strong: bool = False
    ltf_pullback: bool = False
    ltf_continuation: bool = False


def _dir_to_side(d: Any) -> int:
    """Direction enum / str → +1 bull, -1 bear, 0 flat."""
    try:
        name = getattr(d, "name", str(d)).upper()
    except Exception:
        name = str(d).upper()
    if "BULL" in name or name in ("1", "+1"):
        return 1
    if "BEAR" in name or name in ("-1",):
        return -1
    return 0


def classify_path_error(
    *,
    mark_act: int,
    policy_act: int,
    t: int,
    t1: Optional[int],
    t2: Optional[int] = None,
    htf_dir: Any = None,
    ltf_dir: Any = None,
    pullback: Optional[bool] = None,
) -> PathLesson:
    """Map bar → path law for L2L.

    Correct skill: learn to identify LTF pullbacks vs continuations
    **while HTF is strong bull or bear**. Calendar-free.
    """
    ma, pa = int(mark_act), int(policy_act)
    htf = _dir_to_side(htf_dir)
    ltf = _dir_to_side(ltf_dir)
    htf_strong = htf != 0
    # Pullback: HTF clear and LTF opposite (structure.py definition)
    if pullback is None:
        is_pb = htf_strong and ltf != 0 and ltf == -htf
    else:
        is_pb = bool(pullback) and htf_strong
    is_cont = htf_strong and ltf == htf  # LTF confirms HTF

    # --- Error classes first (what not to do) ---
    # Fired when Mark waits: thrash — worst on pullback or weak HTF
    if ma == ACTION_HOLD and pa != ACTION_HOLD:
        return PathLesson(
            path_class="anti_thrash",
            mark_act=ma,
            policy_act=pa,
            weight=20.0 if (is_pb or not htf_strong) else 16.0,
            principle_ids=(
                "wait_is_skill",
                "ltf_pullback_on_htf",
                "htf_gravity",
                "hard_target_quality_over_thrash",
            ),
            htf_strong=htf_strong,
            ltf_pullback=is_pb,
            ltf_continuation=is_cont,
        )
    # Held when Mark fires with HTF: missed continuation
    if ma != ACTION_HOLD and pa == ACTION_HOLD:
        return PathLesson(
            path_class="miss_continuation",
            mark_act=ma,
            policy_act=pa,
            weight=17.0 if is_cont else 14.0,
            principle_ids=(
                "ltf_continuation_on_htf",
                "htf_gravity",
                "learn_not_copy",
            ),
            htf_strong=htf_strong,
            ltf_pullback=is_pb,
            ltf_continuation=is_cont,
        )

    # --- Structure laws (what to learn) ---
    if not htf_strong:
        return PathLesson(
            path_class="htf_not_strong",
            mark_act=ma,
            policy_act=pa,
            weight=12.0 if ma == ACTION_HOLD else 8.0,
            principle_ids=("htf_gravity", "no_invent_without_tide", "wait_is_skill"),
            htf_strong=False,
            ltf_pullback=False,
            ltf_continuation=False,
        )

    if is_pb:
        # LTF pullback while HTF strong → wait / load slingshot
        return PathLesson(
            path_class="ltf_pullback_htf_strong",
            mark_act=ma,
            policy_act=pa,
            weight=14.0 if ma == ACTION_HOLD else 10.0,
            principle_ids=(
                "ltf_pullback_on_htf",
                "wait_is_skill",
                "slingshot_load",
                "htf_gravity",
            ),
            htf_strong=True,
            ltf_pullback=True,
            ltf_continuation=False,
        )

    if is_cont:
        # LTF continuation with HTF → fire / add with tide
        return PathLesson(
            path_class="ltf_continuation_htf_strong",
            mark_act=ma,
            policy_act=pa,
            weight=14.0 if ma != ACTION_HOLD else 9.0,
            principle_ids=(
                "ltf_continuation_on_htf",
                "htf_gravity",
                "ltf_never_votes_side",
                "dominant_trends",
            ),
            htf_strong=True,
            ltf_pullback=False,
            ltf_continuation=True,
        )

    # HTF strong, LTF flat/unclear — still wait for LTF read
    return PathLesson(
        path_class="ltf_pullback_htf_strong" if ma == ACTION_HOLD else "hold_on_spine",
        mark_act=ma,
        policy_act=pa,
        weight=8.0,
        principle_ids=("wait_is_skill", "htf_gravity", "ltf_timing"),
        htf_strong=True,
        ltf_pullback=False,
        ltf_continuation=False,
    )


# Explicit HTF/LTF structure cues for path head (floor-safe: trunk stays child)
# [htf_side, ltf_side, pullback, continuation, htf_strong, aligned, set_id/4, t_phase]
STRUCT_DIM = 8


def pack_struct_feats(
    *,
    htf_dir: Any,
    ltf_dir: Any,
    pullback: Optional[bool],
    set_id: Optional[int],
    t: int,
    n_bars: int = 1500,
) -> np.ndarray:
    """Compact structure vector — path laws learn from this + frozen trunk."""
    htf = float(_dir_to_side(htf_dir))
    ltf = float(_dir_to_side(ltf_dir))
    htf_strong = 1.0 if htf != 0.0 else 0.0
    is_pb = 1.0 if (pullback if pullback is not None else (htf_strong and ltf == -htf and ltf != 0)) else 0.0
    is_cont = 1.0 if (htf_strong and ltf == htf and ltf != 0) else 0.0
    aligned = 1.0 if (htf_strong and ltf == htf) else 0.0
    sid = float(set_id or 0) / 4.0
    phase = float(t) / float(max(n_bars, 1))
    return np.asarray(
        [htf, ltf, is_pb, is_cont, htf_strong, aligned, sid, phase], dtype=np.float32
    )


class PathL2LPolicy(nn.Module):
    """Trunk+act (child floor) + path head on trunk||structure (HTF/LTF laws)."""

    def __init__(self, obs_dim: int = MARK_FULL_DIM, hidden: int = 128, struct_dim: int = STRUCT_DIM):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden = int(hidden)
        self.struct_dim = int(struct_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.obs_dim, hidden),
            nn.Tanh(),
        )
        self.act_head = nn.Linear(hidden, N_ACTIONS)
        # Path head sees structure cues directly → can learn pb/cont without thrashing trunk
        self.path_head = nn.Sequential(
            nn.Linear(hidden + self.struct_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, N_PATH),
        )

    def forward(
        self, obs: torch.Tensor, struct: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        h = self.trunk(obs)
        if struct is None:
            struct = torch.zeros(h.size(0), self.struct_dim, device=h.device, dtype=h.dtype)
        elif struct.dim() == 1:
            struct = struct.unsqueeze(0)
        path_in = torch.cat([h, struct], dim=-1)
        return self.act_head(h), self.path_head(path_in)

    def act_logits(self, obs: torch.Tensor) -> torch.Tensor:
        a, _ = self.forward(obs)
        return a

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray | torch.Tensor,
        *,
        greedy: bool = True,
        struct: np.ndarray | torch.Tensor | None = None,
    ) -> Tuple[int, torch.Tensor]:
        if isinstance(obs, np.ndarray):
            t = torch.as_tensor(obs, dtype=torch.float32)
        else:
            t = obs.float()
        st = None
        if struct is not None:
            st = (
                torch.as_tensor(struct, dtype=torch.float32)
                if isinstance(struct, np.ndarray)
                else struct.float()
            )
        logits = self.forward(t, st)[0].squeeze(0)
        if greedy:
            action = int(torch.argmax(logits).item())
        else:
            action = int(torch.distributions.Categorical(logits=logits).sample().item())
        logp = F.log_softmax(logits, dim=-1)[action]
        return action, logp

    def load_from_channel1(self, state: Dict[str, Any]) -> None:
        """Warm trunk+act from Channel1Policy Sequential weights."""
        try:
            self.trunk[0].weight.data.copy_(state["net.0.weight"])
            self.trunk[0].bias.data.copy_(state["net.0.bias"])
            self.act_head.weight.data.copy_(state["net.2.weight"])
            self.act_head.bias.data.copy_(state["net.2.bias"])
            print("  L2L warm-start: trunk+act from Channel1 embryo", flush=True)
        except Exception as e:
            print(f"  L2L warm-start partial/skip: {e}", flush=True)

    def to_channel1_state(self) -> Dict[str, Any]:
        """Export act path as Channel1Policy state_dict for score_policy / save_policy."""
        return {
            "net.0.weight": self.trunk[0].weight.detach().clone(),
            "net.0.bias": self.trunk[0].bias.detach().clone(),
            "net.2.weight": self.act_head.weight.detach().clone(),
            "net.2.bias": self.act_head.bias.detach().clone(),
        }


def as_channel1(l2l: PathL2LPolicy) -> Channel1Policy:
    pol = Channel1Policy(obs_dim=l2l.obs_dim, hidden=l2l.hidden)
    pol.load_state_dict(l2l.to_channel1_state())
    pol.eval()
    return pol


def load_pattern_memory(path: str = L2L_MEM) -> List[dict]:
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_pattern_memory(row: dict, path: str = L2L_MEM) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def class_boosts_from_memory(mem: List[dict], *, last_n: int = 8) -> Dict[str, float]:
    """Meta: boost laws that dominated recent rejects — prefer pullback/continuation."""
    boost = {c: 1.0 for c in PATH_CLASSES}
    recent = mem[-last_n:] if mem else []
    counts: Counter = Counter()
    for r in recent:
        for c, n in (r.get("class_counts") or {}).items():
            c2 = _CLASS_ALIASES.get(c, c)
            counts[c2] += int(n)
        if r.get("decision") == "REJECT":
            top = r.get("dominant_class")
            if top:
                counts[_CLASS_ALIASES.get(str(top), str(top))] += 3
    if not counts:
        # Default meta prior: pullback + continuation under HTF are the skill
        boost["ltf_pullback_htf_strong"] = 1.4
        boost["ltf_continuation_htf_strong"] = 1.4
        boost["anti_thrash"] = 1.3
        return boost
    total = sum(counts.values()) or 1
    for c, n in counts.items():
        if c in boost:
            boost[c] = 1.0 + 1.5 * (n / total)
    return boost


def collect_l2l_dagger(
    day_map: Dict[str, Any],
    date: str,
    t: float,
    r: float,
    mark: dict,
    policy: Channel1Policy | PathL2LPolicy,
    *,
    class_boost: Optional[Dict[str, float]] = None,
) -> Tuple[List[np.ndarray], List[int], List[int], List[float], List[np.ndarray], Counter]:
    """DAgger walk: Mark act + HTF/LTF path law + structure features."""
    plan = mark.get("plan") or {}
    plan = {int(k): int(v) for k, v in plan.items()}
    t1 = mark.get("t1")
    t2 = mark.get("t2")
    if t1 is None:
        fires = [k for k, v in plan.items() if int(v) != ACTION_HOLD]
        t1 = min(fires) if fires else None
    boost = class_boost or {c: 1.0 for c in PATH_CLASSES}
    for old, new in _CLASS_ALIASES.items():
        if old in boost and new in boost:
            boost[new] = max(float(boost[new]), float(boost[old]))

    day = GoalEquityDay(
        day_map[date],
        target_pct=float(t),
        risk_pct=float(r),
        date_str=date,
        eyes_mode="mark_doctrine",
        mark_soul=True,
        full_obs=True,
        mark_align_policy=True,
    )
    if mark.get("risk_use_frac") not in (None, "dynamic"):
        day._plan_lock_ruf = float(mark["risk_use_frac"])
        day._plan_lock_cap = float(mark["per_trade_cap_pct"])

    xs, y_act, y_cls, ws, structs = [], [], [], [], []
    counts: Counter = Counter()
    set_hits: Counter = Counter()
    prev = 0
    structure_laws = {
        "ltf_pullback_htf_strong",
        "ltf_continuation_htf_strong",
        "htf_not_strong",
        "anti_thrash",
        "miss_continuation",
    }
    n_bars = max(len(day.m1), 1)
    for tb in day.runner.decision_indices():
        if day.dead or day.banked:
            break
        for bt in range(prev, tb):
            if day.dead or day.banked:
                break
            day._mark_bar(bt)
        prev = tb + 1
        if day.dead or day.banked:
            break
        obs = np.asarray(day.observe(tb), np.float32).reshape(-1)
        htf_dir = None
        ltf_dir = None
        is_pb = None
        sid = None
        try:
            perc = day.runner.perceive(int(tb))
            mark_opp = perc.get("mark_opportunity")
            if mark_opp is not None and getattr(mark_opp, "best", None) is not None:
                best = mark_opp.best
                htf_dir = getattr(best, "htf_dir", None) or perc.get("higher")
                ltf_dir = getattr(best, "ltf_dir", None) or perc.get("lower")
                sid = getattr(best, "set_id", None)
                if sid is not None:
                    set_hits[f"set{int(sid)}"] += 1
            else:
                htf_dir = perc.get("higher")
                ltf_dir = perc.get("lower")
            struct = perc.get("structure")
            if struct is not None:
                is_pb = bool(getattr(struct, "pullback", False))
            dec = getattr(day.runner, "last_doctrine", None)
            if dec is not None and getattr(dec, "best_set_id", None) is not None:
                sid = int(dec.best_set_id)
                set_hits[f"set{sid}"] += 1
        except Exception:
            pass
        svec = pack_struct_feats(
            htf_dir=htf_dir,
            ltf_dir=ltf_dir,
            pullback=is_pb,
            set_id=int(sid) if sid is not None else None,
            t=int(tb),
            n_bars=n_bars,
        )
        with torch.no_grad():
            if isinstance(policy, PathL2LPolicy):
                pa, _ = policy.act(obs, greedy=True, struct=svec)
            else:
                pa, _ = policy.act(obs, greedy=True)
            pa = int(pa)
        ma = int(plan.get(int(tb), ACTION_HOLD))
        lesson = classify_path_error(
            mark_act=ma,
            policy_act=pa,
            t=int(tb),
            t1=int(t1) if t1 is not None else None,
            t2=t2,
            htf_dir=htf_dir,
            ltf_dir=ltf_dir,
            pullback=is_pb,
        )
        keep = (
            pa != ma
            or lesson.path_class in structure_laws
            or lesson.ltf_pullback
            or lesson.ltf_continuation
        )
        if keep:
            n = 3 if pa != ma else (2 if lesson.ltf_pullback or lesson.ltf_continuation else 1)
            w = lesson.weight * float(boost.get(lesson.path_class, 1.0))
            if lesson.htf_strong and (lesson.ltf_pullback or lesson.ltf_continuation):
                w *= 1.35
            ci = CLASS_TO_I[lesson.path_class]
            for _ in range(n):
                xs.append(obs.copy())
                y_act.append(ma)
                y_cls.append(ci)
                ws.append(w)
                structs.append(svec.copy())
            counts[lesson.path_class] += n
        day.step_action(tb, pa)
    if set_hits:
        for k, v in set_hits.items():
            counts[f"__tf_{k}"] = int(v)
    return xs, y_act, y_cls, ws, structs, counts


def train_l2l(
    X: np.ndarray,
    y_act: np.ndarray,
    y_cls: np.ndarray,
    *,
    S: Optional[np.ndarray] = None,
    sample_weights: Optional[np.ndarray] = None,
    epochs: int = 24,
    lr: float = 2e-4,
    hidden: int = 128,
    seed: int = 42,
    warm_channel1: Optional[dict] = None,
    kl_anchor_channel1: Optional[dict] = None,
    kl_coef: float = 0.55,
    class_coef: float = 0.55,
    freeze_trunk: bool = False,
    two_phase: bool = True,
) -> Tuple[PathL2LPolicy, List[float], Dict[str, float]]:
    """Train path-class meta + light act.

    two_phase (default True) — surgical L2L for child floor:
      Phase A: freeze trunk+act, train path_head only (class structure).
      Phase B: freeze trunk+path, tiny act updates with high KL to child.
    Prevents act thrash that drops pack 35→32 while still teaching classes.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    pol = PathL2LPolicy(obs_dim=MARK_FULL_DIM, hidden=hidden)
    if warm_channel1 is not None:
        pol.load_from_channel1(warm_channel1)

    anchor = None
    if kl_anchor_channel1 is not None and kl_coef > 0:
        anchor = Channel1Policy(obs_dim=MARK_FULL_DIM, hidden=hidden)
        anchor.load_state_dict(kl_anchor_channel1)
        anchor.eval()
        for p in anchor.parameters():
            p.requires_grad_(False)

    Xt = torch.tensor(X, dtype=torch.float32)
    ya = torch.tensor(y_act, dtype=torch.long)
    yc = torch.tensor(y_cls, dtype=torch.long)
    if S is None:
        St = torch.zeros(len(y_act), STRUCT_DIM, dtype=torch.float32)
    else:
        St = torch.tensor(np.asarray(S, np.float32), dtype=torch.float32)
        if St.dim() == 1:
            St = St.unsqueeze(0)
        if St.size(0) != len(y_act):
            # pad/truncate safety
            pad = torch.zeros(len(y_act), STRUCT_DIM)
            n_copy = min(St.size(0), len(y_act))
            pad[:n_copy] = St[:n_copy]
            St = pad
    if sample_weights is not None:
        sw = np.asarray(sample_weights, np.float32)
        sw = np.maximum(sw, 1e-6)
        sw = sw / float(sw.mean())
        sw_t = torch.tensor(sw, dtype=torch.float32)
    else:
        sw_t = None

    n = len(y_act)
    losses: List[float] = []

    # Path laws where act updates are allowed in phase B
    # NEVER train act on hold_on_spine awards (pack thrash 35→34)
    # Focus: pullback wait + continuation fire under strong HTF
    act_ok_cls = {
        CLASS_TO_I["anti_thrash"],
        CLASS_TO_I["miss_continuation"],
        CLASS_TO_I["ltf_pullback_htf_strong"],
        CLASS_TO_I["ltf_continuation_htf_strong"],
        CLASS_TO_I["htf_not_strong"],
    }
    act_mask_np = np.array([1.0 if int(c) in act_ok_cls else 0.0 for c in y_cls], dtype=np.float32)
    act_mask_t = torch.tensor(act_mask_np, dtype=torch.float32)
    # Inverse-freq class weights for path laws (pullback/continuation balanced vs thrash)
    bc = np.bincount(y_cls.astype(np.int64), minlength=N_PATH).astype(np.float64)
    bc = np.maximum(bc, 1.0)
    cls_bal = (bc.sum() / (float(N_PATH) * bc)).astype(np.float32)
    cls_bal_t = torch.tensor(cls_bal, dtype=torch.float32)
    print(
        f"  L2L path class balance weights: "
        + ", ".join(f"{PATH_CLASSES[i]}={cls_bal[i]:.2f}" for i in range(N_PATH) if bc[i] > 1),
        flush=True,
    )

    def _run_epochs(
        n_ep: int,
        *,
        train_act: bool,
        train_path: bool,
        train_trunk: bool,
        act_w: float,
        cls_w: float,
        kl_w: float,
        lr_e: float,
        tag: str,
        act_path_mask: bool = False,
    ) -> None:
        for p in pol.trunk.parameters():
            p.requires_grad_(train_trunk)
        for p in pol.act_head.parameters():
            p.requires_grad_(train_act)
        for p in pol.path_head.parameters():
            p.requires_grad_(train_path)
        params = [p for p in pol.parameters() if p.requires_grad]
        if not params:
            return
        opt = torch.optim.Adam(params, lr=lr_e)
        print(
            f"  L2L {tag}: act={train_act} path={train_path} trunk={train_trunk} "
            f"ep={n_ep} lr={lr_e} act_w={act_w} cls_w={cls_w} kl_w={kl_w} "
            f"act_path_mask={act_path_mask}",
            flush=True,
        )
        for ep in range(n_ep):
            perm = np.random.permutation(n)
            ep_loss = 0.0
            nb = 0
            for i in range(0, n, 256):
                idx = perm[i : i + 256]
                logits_a, logits_c = pol(Xt[idx], St[idx])
                if sw_t is None:
                    la = F.cross_entropy(logits_a, ya[idx], reduction="none")
                    lc = F.cross_entropy(
                        logits_c, yc[idx], weight=cls_bal_t, reduction="none"
                    )
                    w_a = torch.ones_like(la)
                    w_c = torch.ones_like(lc)
                else:
                    la = F.cross_entropy(logits_a, ya[idx], reduction="none")
                    lc = F.cross_entropy(
                        logits_c, yc[idx], weight=cls_bal_t, reduction="none"
                    )
                    w_a = sw_t[idx]
                    w_c = sw_t[idx]
                if act_path_mask:
                    w_a = w_a * act_mask_t[idx]
                # avoid empty act batch
                if float(w_a.sum().item()) < 1e-6 and act_w > 0:
                    loss_a = torch.tensor(0.0)
                else:
                    loss_a = (la * w_a).sum() / w_a.sum().clamp(min=1e-6)
                loss_c = (lc * w_c).mean()
                loss = float(act_w) * loss_a + float(cls_w) * loss_c
                if anchor is not None and kl_w > 0 and train_act:
                    with torch.no_grad():
                        a_log = anchor(Xt[idx])
                        a_p = F.softmax(a_log, dim=-1)
                        a_lp = F.log_softmax(a_log, dim=-1)
                    n_lp = F.log_softmax(logits_a, dim=-1)
                    kl_bar = (a_p * (a_lp - n_lp)).sum(dim=-1)
                    if act_path_mask:
                        # KL on all bars (protect awards); act CE only on path-error bars
                        kl = kl_bar.mean()
                    else:
                        kl = kl_bar.mean()
                    loss = loss + float(kl_w) * kl
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
                nb += 1
            losses.append(ep_loss / max(nb, 1))
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  l2l {tag} epoch {ep+1}/{n_ep} loss={losses[-1]:.4f}", flush=True)

    if two_phase:
        # A: deeper path head only — FREEZE trunk so act geometry stays child-safe.
        # (Trunk train without perfect act restore caused 35→31/33 pack crater.)
        ep_a = max(20, epochs)
        _run_epochs(
            ep_a,
            train_act=False,
            train_path=True,
            train_trunk=False,
            act_w=0.0,
            cls_w=max(float(class_coef), 1.5),
            kl_w=0.0,
            lr_e=float(lr) * 1.2,
            tag="phaseA_path_only_frozen_trunk",
        )
        pol.eval()
        with torch.no_grad():
            _, lc0 = pol(Xt, St)
            cls0 = float((lc0.argmax(dim=-1).numpy() == y_cls).mean()) if n else 0.0
        if cls0 < 0.60:
            print(
                f"  L2L path_class={cls0:.3f} < 0.60 — extra path+struct (pullback/continuation)",
                flush=True,
            )
            _run_epochs(
                18,
                train_act=False,
                train_path=True,
                train_trunk=False,
                act_w=0.0,
                cls_w=max(float(class_coef), 1.8),
                kl_w=0.0,
                lr_e=float(lr),
                tag="phaseA_extra_path_struct",
            )
            pol.eval()
            with torch.no_grad():
                _, lc1 = pol(Xt, St)
                cls0 = float((lc1.argmax(dim=-1).numpy() == y_cls).mean()) if n else 0.0
            print(f"  L2L path_class after extra={cls0:.3f}", flush=True)
        # B: surgical path-error act only when structure ID is real
        #    trunk frozen → act head still maps child features → floor safe
        if cls0 < 0.55:
            print(
                f"  L2L phaseB SKIP: path_class_match={cls0:.3f} < 0.55 "
                f"(pb/cont structure still weak; NO pack score)",
                flush=True,
            )
        else:
            ep_b = max(6, epochs // 3)
            _run_epochs(
                ep_b,
                train_act=True,
                train_path=False,
                train_trunk=False,
                act_w=0.20,
                cls_w=0.0,
                kl_w=max(float(kl_coef), 1.15),
                lr_e=min(float(lr), 4e-5),
                tag="phaseB_act_pathmask_htf_ltf",
                act_path_mask=True,
            )
    else:
        if freeze_trunk:
            print("  L2L freeze_trunk: act+path heads only", flush=True)
        _run_epochs(
            epochs,
            train_act=True,
            train_path=True,
            train_trunk=not freeze_trunk,
            act_w=1.0,
            cls_w=float(class_coef),
            kl_w=float(kl_coef),
            lr_e=float(lr),
            tag="joint",
        )

    # learn≠copy metrics on train set (act vs class)
    pol.eval()
    with torch.no_grad():
        la, lc = pol(Xt, St)
        pred_a = la.argmax(dim=-1).numpy()
        pred_c = lc.argmax(dim=-1).numpy()
    act_match = float((pred_a == y_act).mean()) if n else 0.0
    cls_match = float((pred_c == y_cls).mean()) if n else 0.0
    gate = check_learn_not_copy(act_match=act_match, topology_match=cls_match, role_map_match=cls_match)
    # path_weak: structure not ready → outer loop must NOT pack-score (floor protect)
    # With struct features, threshold 0.55 for surgical act (stricter quality)
    path_weak = bool(two_phase and cls_match < 0.55)
    metrics = {
        "act_match": act_match,
        "path_class_match": cls_match,
        "learn_not_copy_pass": bool(gate["pass"]),
        "copying": bool(gate["copying"]),
        "final_loss": float(losses[-1]) if losses else 0.0,
        "two_phase": bool(two_phase),
        "path_weak_skip_pack": path_weak,
    }
    print(f"  L2L metrics {metrics}", flush=True)
    return pol, losses, metrics


def run_l2l_cycle(
    *,
    max_rounds: int = 12,
    keep_floor: int = 35,
    freeze_trunk: bool = True,
    kl_coef: float = 0.90,
    class_coef: float = 1.0,
    two_phase: bool = True,
) -> Dict[str, Any]:
    os.makedirs(OUT, exist_ok=True)
    baseline = json.load(open(BASELINE, encoding="utf-8"))
    mark_rows = baseline["rows"]
    floor_clear = int(baseline["policy_clear"])
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
    day_map = {str(d): m1 for d, m1 in days}
    oracle = load_oracle()
    policy = load_policy(CKPT)
    dials = clip_streak_dials(default_streak_dials())
    mem = load_pattern_memory()

    # Pin timeframe law at start of every climb
    try:
        assert_mark_sets_law()
        print("L2L MARK SETS LAW (LTF first | HTF confirm last two):", flush=True)
        for row in mark_sets_law_table():
            print(
                f"  set{row['set_id']} {row['name']}: "
                f"LTF={row['ltf_entry']} | HTF={'+'.join(row['htf_confirm'])} "
                f"→ LTF={row['ltf_job']} · HTF={row['htf_job']}",
                flush=True,
            )
        print(
            "  skill: identify LTF pullback vs continuation while HTF strong bull|bear "
            "(scan all 4 sets)",
            flush=True,
        )
    except Exception as e:
        print(f"  WARN MARK SETS LAW check: {e}", flush=True)

    print("L2L path cycle — score…", flush=True)
    best = score_policy(policy, day_map, mark_rows)
    print(
        f"START same={best['same_outcome']} mwt={best['mark_would_take']} "
        f"breach={best['n_breach']}",
        flush=True,
    )
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    live_floor = max(keep_floor, int(best["same_outcome"]))
    cycles = []

    for rnd in range(1, max_rounds + 1):
        if best["same_outcome"] >= 50 and best["n_breach"] == 0:
            break
        policy.load_state_dict(best_state)
        boost = class_boosts_from_memory(mem)
        print(f"\n===== L2L {rnd}/{max_rounds} class_boost={ {k: round(v,2) for k,v in boost.items() if v>1.05} } =====", flush=True)

        mwt = sorted(
            [r for r in best["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"],
            key=lambda r: float(r.get("policy_pnl") or 0),
        )
        awards = [r for r in best["rows"] if r["miss_class"] == "AWARD"]
        if not mwt:
            break
        # focus rotate; sample 3–4 MWT for class diversity (meta), not pack thrash
        focus = mwt[(rnd - 1) % len(mwt)]
        support = [mwt[i % len(mwt)] for i in range(rnd, rnd + 3) if mwt[i % len(mwt)] is not focus]
        # de-dupe by date
        seen_d = {focus["date"]}
        targets = [focus]
        for row in support + mwt:
            if row["date"] in seen_d:
                continue
            targets.append(row)
            seen_d.add(row["date"])
            if len(targets) >= 4:
                break
        print(f"  focus={focus['date']} +{len(targets)-1} support for class coverage", flush=True)

        xs, ya, yc, ws, ss = [], [], [], [], []
        all_counts: Counter = Counter()
        for row in targets:
            mark = get_plan(
                oracle, day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"])
            )
            a, b, c, d, svecs, cnt = collect_l2l_dagger(
                day_map,
                row["date"],
                float(row["target_pct"]),
                float(row["risk_pct"]),
                mark,
                policy,
                class_boost=boost,
            )
            xs.extend(a)
            ya.extend(b)
            yc.extend(c)
            ws.extend(d)
            ss.extend(svecs)
            all_counts.update(cnt)
        # award protect (act only as HOLD/self — path class hold_on_spine)
        for row in awards[:22]:
            a, b, c = award_self(
                day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"]), policy
            )
            for o, act, w in zip(a, b, c):
                xs.append(o)
                ya.append(int(act))
                yc.append(CLASS_TO_I["hold_on_spine"])
                ws.append(float(w) * 2.0)
                ss.append(np.zeros(STRUCT_DIM, dtype=np.float32))

        if len(ya) < 40:
            print("  too few L2L labels", flush=True)
            continue
        X = np.stack(xs)
        y_act = np.asarray(ya, np.int64)
        y_cls = np.asarray(yc, np.int64)
        w = np.asarray(ws, np.float32)
        S = np.stack(ss) if ss else np.zeros((len(ya), STRUCT_DIM), np.float32)
        print(
            f"  n={len(y_act)} classes={ {k:v for k,v in dict(all_counts).items() if not str(k).startswith('__')} } "
            f"dir={int((y_act!=0).sum())} hold={int((y_act==0).sum())} "
            f"struct_dim={S.shape[-1]}",
            flush=True,
        )

        # Intelligent climb: path head on trunk||HTF/LTF struct (trunk frozen) → floor safe
        l2l, _, metrics = train_l2l(
            X,
            y_act,
            y_cls,
            S=S,
            sample_weights=w,
            epochs=24,
            lr=2.0e-4,
            warm_channel1=best_state,
            kl_anchor_channel1=best_state,
            kl_coef=float(kl_coef),
            class_coef=float(class_coef),
            freeze_trunk=bool(freeze_trunk),
            two_phase=bool(two_phase),
            seed=700 + rnd,
        )
        # hard gates: copying OR path_weak → no pack score (protect child floor)
        if metrics.get("copying") or metrics.get("path_weak_skip_pack"):
            reason = "learn_not_copy" if metrics.get("copying") else "path_weak_skip_pack"
            print(
                f"  {reason} — REJECT without pack score thrash "
                f"(path={metrics.get('path_class_match'):.3f})",
                flush=True,
            )
            row_mem = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "round": rnd,
                "decision": "REJECT",
                "reason": reason,
                "class_counts": {
                    k: v for k, v in dict(all_counts).items() if not str(k).startswith("__")
                },
                "dominant_class": next(
                    (c for c, _ in all_counts.most_common() if not str(c).startswith("__")),
                    None,
                ),
                "metrics": metrics,
                "same": best["same_outcome"],
            }
            append_pattern_memory(row_mem)
            mem.append(row_mem)
            cycles.append(row_mem)
            continue

        pol2 = as_channel1(l2l)
        post = score_policy(pol2, day_map, mark_rows)
        focus_ok = any(
            r["date"] == focus["date"] and r["policy_award"] for r in post["rows"]
        )
        print(
            f"  focus {focus['date']} award={focus_ok} POST same={post['same_outcome']} "
            f"mwt={post['mark_would_take']} breach={post['n_breach']}",
            flush=True,
        )

        if post["same_outcome"] < best["same_outcome"] - 3:
            print("  pack crater — REJECT", flush=True)
            decision = "REJECT"
        else:
            keep = (
                post["n_breach"] == 0
                and post["policy_clear"] >= floor_clear
                and post["same_outcome"] >= live_floor
                and (
                    post["same_outcome"] > best["same_outcome"]
                    or (
                        focus_ok
                        and post["same_outcome"] >= best["same_outcome"]
                        and post["policy_clear"] >= best["policy_clear"]
                    )
                )
            )
            decision = "KEEP" if keep else "REJECT"

        if decision == "KEEP":
            best = post
            best_state = {k: v.detach().clone() for k, v in pol2.state_dict().items()}
            live_floor = max(live_floor, int(post["same_outcome"]))
            save_policy(pol2, note=f"l2l_KEEP_r{rnd}_{focus['date']}", dials=dials)
            torch.save(
                {
                    "tag": "mark_shadow_l2l_v1",
                    "state_dict_channel1": best_state,
                    "l2l_state": l2l.state_dict(),
                    "path_classes": PATH_CLASSES,
                    "same_outcome": post["same_outcome"],
                    "proven_touched": False,
                    "method": "learn_to_learn_path",
                    "metrics": metrics,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                },
                L2L_CKPT,
            )
            with open(BEST, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "same_outcome": post["same_outcome"],
                        "policy_clear": post["policy_clear"],
                        "mwt": post["mark_would_take"],
                        "breach": post["n_breach"],
                        "source": f"l2l_KEEP_{focus['date']}",
                        "l2l": True,
                        "stage": "teen" if post["same_outcome"] > 35 else "child",
                        "child_frozen_sha256": "9BDCEAAE3B282DA1548F6C58E55F5935AED5ECF5720EC95C4913CE17F06FD555",
                        "child_backup": "CHILD_STAGE_same35_mark_clone_full_obs.pt",
                        "l2l_rules": "L2L_RULES__binding.json",
                        "method": "learn_to_learn_path",
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    f,
                    indent=2,
                )
            print(f"  KEEP best_same={best['same_outcome']}", flush=True)
            try:
                with open(os.path.join(OUT, "WHAT_WORKS__GOAL.md"), "a", encoding="utf-8") as wf:
                    wf.write(
                        f"| KEEP live | **{best['same_outcome']}** | {best['mark_would_take']} | "
                        f"{best['n_breach']} | l2l path-class {focus['date']} |\n"
                    )
            except OSError:
                pass
        else:
            print("  REJECT", flush=True)

        dom = all_counts.most_common(1)[0][0] if all_counts else None
        row_mem = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "round": rnd,
            "focus": focus["date"],
            "focus_ok": focus_ok,
            "decision": decision,
            "same": post["same_outcome"],
            "mwt": post["mark_would_take"],
            "breach": post["n_breach"],
            "best_same": best["same_outcome"],
            "class_counts": dict(all_counts),
            "dominant_class": dom,
            "metrics": metrics,
            "class_boost_used": {k: round(v, 3) for k, v in boost.items()},
        }
        append_pattern_memory(row_mem)
        mem.append(row_mem)
        cycles.append(row_mem)

        # write human report
        with open(L2L_REPORT, "w", encoding="utf-8") as f:
            f.write(
                f"# L2L path cycle report\n\n"
                f"- best_same: {best['same_outcome']}\n"
                f"- last decision: {decision}\n"
                f"- dominant_class: {dom}\n"
                f"- learn_not_copy: {metrics}\n"
                f"- class_counts: {dict(all_counts)}\n"
                f"- memory_boost next: {class_boosts_from_memory(mem)}\n"
            )
        with open(L2L_STATE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "best_same": best["same_outcome"],
                    "best_mwt": best["mark_would_take"],
                    "cycles": cycles,
                    "path_classes": PATH_CLASSES,
                    "next_class_boost": class_boosts_from_memory(mem),
                },
                f,
                indent=2,
            )

    summary = {
        "best_same": best["same_outcome"],
        "best_mwt": best["mark_would_take"],
        "best_breach": best["n_breach"],
        "cycles": cycles,
        "method": "learn_to_learn_path",
    }
    print(
        f"DONE L2L best same={summary['best_same']} mwt={summary['best_mwt']}",
        flush=True,
    )
    return summary


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Learn-to-learn path classes (not day memos)")
    ap.add_argument("--max-rounds", type=int, default=10)
    ap.add_argument("--keep-floor", type=int, default=35)
    ap.add_argument("--freeze-trunk", action="store_true", default=True)
    ap.add_argument("--no-freeze-trunk", action="store_true")
    ap.add_argument("--kl-coef", type=float, default=0.90)
    ap.add_argument("--class-coef", type=float, default=1.0)
    ap.add_argument("--two-phase", action="store_true", default=True)
    ap.add_argument("--no-two-phase", action="store_true")
    args = ap.parse_args()
    ft = False if args.no_freeze_trunk else True
    tp = False if args.no_two_phase else True
    run_l2l_cycle(
        max_rounds=args.max_rounds,
        keep_floor=args.keep_floor,
        freeze_trunk=ft,
        kl_coef=args.kl_coef,
        class_coef=args.class_coef,
        two_phase=tp,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
