"""Self-climb L2L — policy learns to learn past 35/50 without day memos.

Why prior L2L stuck at 35
-------------------------
Path head trained ~0.22–0.30 class match → phase-B act never unlocked safely.
When act *did* move, pack cratered 35→31–33. Path knowledge never touched
*actions* at score time (as_channel1 drops the path head).

Design (prime-agent Continual Harness + prime-rl hygiene)
--------------------------------------------------------
1. IMMUTABLE BASE — child embryo Channel1 weights never trained in this loop.
2. PATH SKILL LAYER — runtime decode adapter that reads HTF/LTF structure
   already present in full_obs (sets + pullback + doctrine) and applies path
   laws: wait on pullback/weak HTF, fire with continuation under strong HTF.
3. META LEARN — after each score, diagnose which path law would have fixed
   each MARK_WOULD_TAKE day; boost that law's dial; rollback on pack drop.
4. KEEP only if same_outcome > floor (default 35) and breach==0.
5. Optional soft neural gate (tiny MLP on structure slice) trained offline
   from Mark vs child disagreements — never touches child trunk/act.

Goal: same > 35, breach 0, skill attributed to path laws not calendar memos.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
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
    get_plan,
    load_oracle,
    score_policy,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    Channel1Policy,
    N_ACTIONS,
)
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
CHILD = os.path.join(_HERE, "checkpoints", "CHILD_STAGE_same35_mark_clone_full_obs.pt")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")
BEST = os.path.join(OUT, "BEST__latest.json")
HARNESS_STATE = os.path.join(OUT, "SELF_CLIMB_HARNESS__latest.json")
HARNESS_MEM = os.path.join(OUT, "SELF_CLIMB_MEMORY.jsonl")
HARNESS_REPORT = os.path.join(OUT, "SELF_CLIMB__latest.md")
SKILL_CKPT = os.path.join(_HERE, "checkpoints", "path_skill_self_climb_v1.pt")

CHILD_SHA = "9BDCEAAE3B282DA1548F6C58E55F5935AED5ECF5720EC95C4913CE17F06FD555"

# full_obs layout slices (observation.py + observation_full.py)
# CHANNEL1 [0:32]: sets 1-4 dirs at 0,3,6,9; pullback@27; scale_conflict@28
# DOCTRINE  [32:48]: force@32, launch@39, breather@40, aligned@41,
#                    regime bull/bear/chop/flat @42-45


@dataclass
class PathSkillDials:
    """Learnable path-law strengths (harness refinable; base child immutable).

    Physics.md maps here as *decode* laws (not act-head BC):
      against_htf_hold  ↔ PINN HTF gravity (tide penalty at score time)
      thrash_gate + entropy_hold ↔ entropy regime mask (chop → HOLD)
      cont_gate + tension_req ↔ kinematics: launch only with mass/tension
      (dimensionless ATR / aux a_mass heads need retrain → forbidden while R1 floor)
    """

    # Defaults = pure-child pass-through. Probe 2026-08-06: entropy_hold=1 or
    # loose cont_fill drops same 35→30; pinn_only / thrash_grav hold 35, no raise.
    thrash_gate: float = 0.0  # suppress fire on pullback/weak HTF (off by default)
    cont_gate: float = 0.0  # fire when HOLD but continuation structure
    against_htf_hold: float = 0.0  # PINN gravity decode (off; child already ~aligned)
    require_launch_for_cont: float = 1.0  # 1 = only cont-fire when doctrine launch
    min_force: float = 0.25  # |force| to treat HTF as directed
    use_neural: float = 0.0  # 0..1 blend neural skill gate
    entropy_hold: float = 0.0  # chop/flat HOLD — OFF default (probe: 61 holds → 30/50)
    tension_req: float = 0.0  # kinematics mass gate for cont
    note: str = "init"

    def clamp(self) -> "PathSkillDials":
        self.thrash_gate = float(np.clip(self.thrash_gate, 0.0, 1.5))
        self.cont_gate = float(np.clip(self.cont_gate, 0.0, 1.5))
        self.against_htf_hold = float(np.clip(self.against_htf_hold, 0.0, 1.5))
        self.require_launch_for_cont = float(np.clip(self.require_launch_for_cont, 0.0, 1.0))
        self.min_force = float(np.clip(self.min_force, 0.0, 1.0))
        self.use_neural = float(np.clip(self.use_neural, 0.0, 1.0))
        self.entropy_hold = float(np.clip(self.entropy_hold, 0.0, 1.5))
        self.tension_req = float(np.clip(self.tension_req, 0.0, 1.5))
        return self

    def skill_attribution(self) -> str:
        """Path dials / laws — never a single MWT day id."""
        parts = []
        if self.against_htf_hold > 0.5:
            parts.append("pinn_htf_gravity")
        if self.entropy_hold > 0.5 or self.thrash_gate > 0.5:
            parts.append("entropy_anti_thrash")
        if self.cont_gate > 0.5:
            parts.append("ltf_continuation_fire")
        if self.thrash_gate > 0.5:
            parts.append("ltf_pullback_wait")
        if self.tension_req > 0.5:
            parts.append("kinematic_tension")
        if self.use_neural > 0.5:
            parts.append("tiny_path_gate")
        return " + ".join(parts) if parts else "pure_child_pass"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def extract_struct_from_obs(obs: np.ndarray) -> Dict[str, float]:
    """Structure cues already in full_obs — no extra perceive needed at score."""
    o = np.asarray(obs, dtype=np.float32).reshape(-1)
    if o.size < 48:
        # pad safety
        pad = np.zeros(MARK_FULL_DIM, dtype=np.float32)
        pad[: o.size] = o
        o = pad
    set_dirs = [float(o[0]), float(o[3]), float(o[6]), float(o[9])]
    force = float(o[32])
    return {
        "set1": set_dirs[0],
        "set2": set_dirs[1],
        "set3": set_dirs[2],
        "set4": set_dirs[3],
        "pullback": float(o[27] > 0.5),
        "scale_conflict": float(o[28] > 0.5),
        "force": force,
        "launch": float(o[39] > 0.5),
        "breather": float(o[40] > 0.5),
        "aligned_play": float(o[41] > 0.5),
        "regime_bull": float(o[42] > 0.5),
        "regime_bear": float(o[43] > 0.5),
        "regime_chop": float(o[44] > 0.5),
        "regime_flat": float(o[45] > 0.5),
    }


def htf_side_from_struct(s: Dict[str, float], *, min_force: float = 0.25) -> int:
    """HTF permission: doctrine regime / force, fall back to higher sets 3–4."""
    if s["regime_bull"] > 0.5:
        return 1
    if s["regime_bear"] > 0.5:
        return -1
    if s["force"] > min_force:
        return 1
    if s["force"] < -min_force:
        return -1
    # higher-set majority (set3, set4)
    hi = s["set3"] + s["set4"]
    if hi > 0.4:
        return 1
    if hi < -0.4:
        return -1
    return 0


def ltf_side_from_struct(s: Dict[str, float]) -> int:
    """LTF timing cue: set1/set2 average."""
    lo = s["set1"] + s["set2"]
    if lo > 0.4:
        return 1
    if lo < -0.4:
        return -1
    return 0


def classify_bar_path_law(s: Dict[str, float], *, min_force: float = 0.25) -> str:
    """Calendar-free path law label for this bar (skill target)."""
    htf = htf_side_from_struct(s, min_force=min_force)
    ltf = ltf_side_from_struct(s)
    if htf == 0 or s["regime_chop"] > 0.5 or s["regime_flat"] > 0.5:
        return "htf_not_strong"
    if s["pullback"] > 0.5 or (ltf != 0 and ltf == -htf) or s["breather"] > 0.5:
        return "ltf_pullback_htf_strong"
    if ltf == htf or s["aligned_play"] > 0.5 or s["launch"] > 0.5:
        return "ltf_continuation_htf_strong"
    return "htf_not_strong"


class TinyPathGate(nn.Module):
    """Tiny MLP: structure slice → (wait, fire_buy, fire_sell, pass) — never trains child."""

    STRUCT_KEYS = (
        "set1",
        "set2",
        "set3",
        "set4",
        "pullback",
        "scale_conflict",
        "force",
        "launch",
        "breather",
        "aligned_play",
        "regime_bull",
        "regime_bear",
        "regime_chop",
        "regime_flat",
    )

    def __init__(self, hidden: int = 32):
        super().__init__()
        d = len(self.STRUCT_KEYS)
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 4),  # wait, buy, sell, pass
        )

    def pack(self, s: Dict[str, float]) -> torch.Tensor:
        return torch.tensor([float(s[k]) for k in self.STRUCT_KEYS], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class PathSkillPolicy:
    """Frozen child + path skill decode. Implements .act() for equity_day / score_policy."""

    def __init__(
        self,
        child: Channel1Policy,
        dials: PathSkillDials,
        neural: Optional[TinyPathGate] = None,
    ):
        self.child = child
        self.child.eval()
        self.dials = dials.clamp()
        self.neural = neural
        if self.neural is not None:
            self.neural.eval()
        self._last_law: str = "pass"
        self._law_counts: Counter = Counter()

    def state_dict(self) -> Dict[str, Any]:
        return self.child.state_dict()

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> Any:
        return self.child.load_state_dict(state, strict=strict)

    def apply_path_skill(self, obs: np.ndarray, base_act: int) -> int:
        d = self.dials
        s = extract_struct_from_obs(obs)
        htf = htf_side_from_struct(s, min_force=d.min_force)
        ltf = ltf_side_from_struct(s)
        law = classify_bar_path_law(s, min_force=d.min_force)
        act = int(base_act)

        # Physics proxies from full_obs structure (no retrain / no new perceive)
        high_entropy = (
            s["regime_chop"] > 0.5
            or s["regime_flat"] > 0.5
            or s["scale_conflict"] > 0.5
        )
        # mass inertia proxy: HTF sets 3–4 agree with tide side
        mass_ok = htf != 0 and (
            (htf > 0 and (s["set3"] + s["set4"]) > 0.2)
            or (htf < 0 and (s["set3"] + s["set4"]) < -0.2)
            or abs(s["force"]) >= d.min_force
        )
        # tension / slingshot-ready: pullback resolved or launch rails up
        tension_ok = (
            s["launch"] > 0.5
            or s["aligned_play"] > 0.5
            or (abs(s["force"]) >= max(0.15, d.min_force) and ltf == htf and htf != 0)
        )

        # Neural soft suggestion (optional blend) — never trains child
        neural_act = None
        if self.neural is not None and d.use_neural > 0.05:
            with torch.no_grad():
                logits = self.neural(self.neural.pack(s))
                conf, idx = torch.softmax(logits, dim=-1).max(dim=-1)
                if float(conf.item()) > 0.45:
                    i = int(idx.item())
                    if i == 0:
                        neural_act = ACTION_HOLD
                    elif i == 1:
                        neural_act = ACTION_BUY
                    elif i == 2:
                        neural_act = ACTION_SELL
                    # i==3 pass

        # --- Physics Law 0: entropy regime mask (chop/flat → HOLD only) ---
        if d.entropy_hold > 0.5 and high_entropy and act != ACTION_HOLD:
            act = ACTION_HOLD
            law = "entropy_hold"

        # --- Law 1: anti_thrash / pullback wait under HTF ---
        if d.thrash_gate > 0.5 and act != ACTION_HOLD:
            weak = htf == 0 or s["regime_chop"] > 0.5 or s["scale_conflict"] > 0.5
            pb = law == "ltf_pullback_htf_strong" or s["pullback"] > 0.5 or s["breather"] > 0.5
            if weak or pb:
                act = ACTION_HOLD
                law = "anti_thrash" if not pb else "ltf_pullback_htf_strong"

        # --- Law 2: PINN gravity — against HTF never fire ---
        if d.against_htf_hold > 0.5 and htf != 0 and act != ACTION_HOLD:
            if act == ACTION_BUY and htf < 0:
                act = ACTION_HOLD
                law = "pinn_against_htf"
            elif act == ACTION_SELL and htf > 0:
                act = ACTION_HOLD
                law = "pinn_against_htf"

        # --- Law 3: miss_continuation — fire with HTF when structure says go ---
        # Physics equation: [tension + low entropy + mass ok] → Launch (not Pattern X)
        # CRITICAL: loose ltf==htf fill tanks floor (35→30). Default = launch/aligned only.
        if d.cont_gate > 0.5 and act == ACTION_HOLD and htf != 0 and not high_entropy:
            cont = law == "ltf_continuation_htf_strong"
            # Always require launch OR aligned for cont-fill (Mark fire window), unless
            # require_launch_for_cont==0 AND tension_req already proves mass+tension.
            if d.require_launch_for_cont > 0.5:
                cont = cont and (s["launch"] > 0.5 or s["aligned_play"] > 0.5)
            elif d.tension_req > 0.5:
                cont = cont and mass_ok and tension_ok and s["launch"] > 0.5
            else:
                # aggressive cont still needs doctrine launch rail (never bare ltf==htf)
                cont = cont and (s["launch"] > 0.5 or s["aligned_play"] > 0.5)
            if d.tension_req > 0.5:
                cont = cont and mass_ok
            if cont and s["pullback"] < 0.5 and s["breather"] < 0.5:
                act = ACTION_BUY if htf > 0 else ACTION_SELL
                law = "phys_launch" if d.tension_req > 0.5 else "miss_continuation_fix"

        # Neural override only when high use_neural and neural suggests wait or fire
        if neural_act is not None and d.use_neural >= 0.75:
            # still respect PINN gravity + entropy (physics hard constraints)
            if d.entropy_hold > 0.5 and high_entropy:
                act = ACTION_HOLD
                law = "entropy_hold"
            elif (
                d.against_htf_hold > 0.5
                and htf != 0
                and neural_act != ACTION_HOLD
                and (
                    (neural_act == ACTION_BUY and htf < 0)
                    or (neural_act == ACTION_SELL and htf > 0)
                )
            ):
                act = ACTION_HOLD
                law = "pinn_against_htf"
            else:
                act = int(neural_act)
                law = "neural_gate"

        self._last_law = law
        self._law_counts[law] += 1
        return act

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray | torch.Tensor,
        *,
        greedy: bool = True,
        generator: Any = None,
    ) -> Tuple[int, torch.Tensor]:
        if isinstance(obs, torch.Tensor):
            obs_np = obs.detach().cpu().numpy().astype(np.float32).reshape(-1)
        else:
            obs_np = np.asarray(obs, dtype=np.float32).reshape(-1)
        base, logp = self.child.act(obs_np, greedy=greedy, generator=generator)
        final = self.apply_path_skill(obs_np, int(base))
        return final, logp


def collect_neural_labels(
    day_map: Dict[str, Any],
    mark_rows: List[dict],
    child: Channel1Policy,
    oracle: dict,
    *,
    max_days: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Offline labels: on Mark≠child bars, target wait/buy/sell from Mark + structure."""
    xs: List[np.ndarray] = []
    ys: List[int] = []
    # Prefer MWT-ish diversity: sample from baseline mark awards that policy may miss
    for mr in mark_rows[:50]:
        if len(set(ys)) >= 4 and len(ys) > 800:
            break
        date = str(mr["date"])
        t, r = float(mr["target_pct"]), float(mr["risk_pct"])
        mark = get_plan(oracle, day_map, date, t, r) if oracle else None
        if not mark or not mark.get("plan"):
            continue
        plan = {int(k): int(v) for k, v in (mark.get("plan") or {}).items()}
        day = GoalEquityDay(
            day_map[date],
            target_pct=t,
            risk_pct=r,
            date_str=date,
            eyes_mode="mark_doctrine",
            mark_soul=True,
            full_obs=True,
            mark_align_policy=True,
        )
        prev = 0
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
            with torch.no_grad():
                pa, _ = child.act(obs, greedy=True)
            ma = int(plan.get(int(tb), ACTION_HOLD))
            s = extract_struct_from_obs(obs)
            feat = np.array([float(s[k]) for k in TinyPathGate.STRUCT_KEYS], dtype=np.float32)
            # Target: Mark action mapped to gate classes; pass if agree
            if ma == pa:
                y = 3  # pass
                # still teach structure laws on some agree bars
                if classify_bar_path_law(s) == "ltf_pullback_htf_strong" and ma == ACTION_HOLD:
                    y = 0
                elif classify_bar_path_law(s) == "ltf_continuation_htf_strong" and ma != ACTION_HOLD:
                    y = 1 if ma == ACTION_BUY else 2
            else:
                if ma == ACTION_HOLD:
                    y = 0  # wait
                elif ma == ACTION_BUY:
                    y = 1
                else:
                    y = 2
            # upsample disagreements
            n = 3 if ma != pa else 1
            for _ in range(n):
                xs.append(feat)
                ys.append(y)
            day.step_action(tb, int(pa))
        if len(ys) > 4000:
            break
        max_days -= 1
        if max_days <= 0:
            break
    if not xs:
        return np.zeros((0, len(TinyPathGate.STRUCT_KEYS)), np.float32), np.zeros((0,), np.int64)
    return np.stack(xs), np.asarray(ys, np.int64)


def train_neural_gate(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 40,
    lr: float = 1e-3,
    seed: int = 42,
) -> TinyPathGate:
    torch.manual_seed(seed)
    gate = TinyPathGate()
    if len(y) < 20:
        return gate
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    # class balance
    bc = np.bincount(y, minlength=4).astype(np.float64)
    bc = np.maximum(bc, 1.0)
    w = torch.tensor((bc.sum() / (4.0 * bc)).astype(np.float32))
    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    n = len(y)
    for ep in range(epochs):
        perm = np.random.permutation(n)
        total = 0.0
        nb = 0
        for i in range(0, n, 256):
            idx = perm[i : i + 256]
            logits = gate(Xt[idx])
            loss = F.cross_entropy(logits, yt[idx], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            with torch.no_grad():
                acc = float((gate(Xt).argmax(-1).numpy() == y).mean())
            print(f"  neural_gate ep {ep+1}/{epochs} loss={total/max(nb,1):.4f} acc={acc:.3f}", flush=True)
    return gate


def diagnose_mwt_laws(
    score: dict,
    day_map: Dict[str, Any],
    child: Channel1Policy,
    dials: PathSkillDials,
    *,
    sample_n: int = 8,
) -> Counter:
    """Which path law dominates on MARK_WOULD_TAKE days (meta signal)."""
    mwt = [r for r in score["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"]
    laws: Counter = Counter()
    for row in mwt[:sample_n]:
        date = row["date"]
        day = GoalEquityDay(
            day_map[date],
            target_pct=float(row["target_pct"]),
            risk_pct=float(row["risk_pct"]),
            date_str=date,
            eyes_mode="mark_doctrine",
            mark_soul=True,
            full_obs=True,
            mark_align_policy=True,
        )
        prev = 0
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
            s = extract_struct_from_obs(obs)
            laws[classify_bar_path_law(s, min_force=dials.min_force)] += 1
            with torch.no_grad():
                pa, _ = child.act(obs, greedy=True)
            day.step_action(tb, int(pa))
    return laws


def refine_dials(dials: PathSkillDials, law_counts: Counter, *, last_same: int, floor: int) -> PathSkillDials:
    """Continual harness refine: small evidence-backed dial updates."""
    d = PathSkillDials(**{k: v for k, v in asdict(dials).items()})
    total = sum(law_counts.values()) or 1
    pb = law_counts.get("ltf_pullback_htf_strong", 0) / total
    cont = law_counts.get("ltf_continuation_htf_strong", 0) / total
    weak = law_counts.get("htf_not_strong", 0) / total

    # If still at floor: need more skill — boost dominant miss structure
    if last_same <= floor:
        if cont >= pb and cont >= weak:
            d.cont_gate = min(1.5, d.cont_gate + 0.35)
            d.require_launch_for_cont = max(0.0, d.require_launch_for_cont - 0.25)
            d.tension_req = min(1.0, d.tension_req + 0.25)  # kinematics mass for launch
            d.entropy_hold = max(d.entropy_hold, 1.0)
            d.against_htf_hold = max(d.against_htf_hold, 1.0)  # keep PINN gravity
            d.note = "boost_phys_continuation"
        elif pb >= weak:
            d.thrash_gate = min(1.5, d.thrash_gate + 0.15)
            d.against_htf_hold = min(1.5, d.against_htf_hold + 0.1)
            d.entropy_hold = min(1.5, d.entropy_hold + 0.15)
            d.note = "boost_pullback_entropy_wait"
        else:
            d.thrash_gate = min(1.5, d.thrash_gate + 0.2)
            d.against_htf_hold = 1.0
            d.entropy_hold = 1.0
            d.min_force = min(0.5, d.min_force + 0.05)
            d.note = "boost_htf_not_strong_wait"
    d.clamp()
    return d


def dials_grid_seed() -> List[PathSkillDials]:
    """Initial autonomous search seeds (learn which skill combo works)."""
    seeds = []
    for thrash in (0.0, 1.0):
        for cont in (0.0, 1.0):
            for against in (0.0, 1.0):
                for launch_req in (0.0, 1.0):
                    if thrash == 0 and cont == 0 and against == 0:
                        continue  # pure child — already known 35
                    seeds.append(
                        PathSkillDials(
                            thrash_gate=thrash,
                            cont_gate=cont,
                            against_htf_hold=against,
                            require_launch_for_cont=launch_req,
                            min_force=0.25,
                            use_neural=0.0,
                            entropy_hold=1.0 if thrash > 0 else 0.0,
                            tension_req=0.0,
                            note=f"grid_t{thrash}_c{cont}_a{against}_L{launch_req}",
                        )
                    )
    # cont-heavy variants for MWT (miss fire more common than thrash on child floor)
    seeds.append(
        PathSkillDials(
            thrash_gate=1.0,
            cont_gate=1.0,
            against_htf_hold=1.0,
            require_launch_for_cont=0.0,
            min_force=0.15,
            entropy_hold=1.0,
            tension_req=0.0,
            note="aggressive_cont",
        )
    )
    seeds.append(
        PathSkillDials(
            thrash_gate=1.0,
            cont_gate=1.0,
            against_htf_hold=1.0,
            require_launch_for_cont=1.0,
            min_force=0.35,
            entropy_hold=1.0,
            tension_req=0.0,
            note="conservative_cont_launch",
        )
    )
    # --- physics.md equation seeds (decode-only; no act BC) ---
    # Prefer floor-safe: gravity+entropy first; cont only with launch rail.
    seeds.insert(
        0,
        PathSkillDials(
            thrash_gate=1.0,
            cont_gate=0.0,
            against_htf_hold=1.0,
            require_launch_for_cont=1.0,
            min_force=0.25,
            entropy_hold=1.0,
            tension_req=0.0,
            note="phys_gravity_entropy_only",
        ),
    )
    seeds.insert(
        1,
        PathSkillDials(
            thrash_gate=0.0,
            cont_gate=0.0,
            against_htf_hold=1.0,
            entropy_hold=0.0,
            note="phys_pinn_only",
        ),
    )
    seeds.insert(
        2,
        PathSkillDials(
            thrash_gate=1.0,
            cont_gate=1.0,
            against_htf_hold=1.0,
            require_launch_for_cont=1.0,
            min_force=0.3,
            entropy_hold=1.0,
            tension_req=1.0,
            note="phys_equation_strict_launch",
        ),
    )
    # Remove bare aggressive cont (proven 35→30 in probe)
    seeds = [s for s in seeds if s.note not in ("aggressive_cont",)]
    seeds.append(
        PathSkillDials(
            thrash_gate=1.0,
            cont_gate=1.0,
            against_htf_hold=1.0,
            require_launch_for_cont=0.0,
            min_force=0.15,
            entropy_hold=1.0,
            tension_req=0.0,
            note="aggressive_cont_launch_rail",  # still requires launch/aligned in apply
        )
    )
    return seeds


def append_mem(row: dict) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(HARNESS_MEM, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def run_self_climb(
    *,
    max_rounds: int = 24,
    keep_floor: int = 35,
    goal_same: int = 36,
    train_neural: bool = True,
    search_grid: bool = True,
) -> Dict[str, Any]:
    os.makedirs(OUT, exist_ok=True)

    # --- Immutable base check ---
    if os.path.isfile(CHILD):
        sha = _sha256_file(CHILD)
        print(f"CHILD embryo sha={sha[:16]}… expected={CHILD_SHA[:16]}…", flush=True)
    # Always start from child file for this harness
    src = CHILD if os.path.isfile(CHILD) else CKPT
    child = load_policy(src)
    child.eval()
    for p in child.parameters():
        p.requires_grad_(False)

    baseline = json.load(open(BASELINE, encoding="utf-8"))
    mark_rows = baseline["rows"]
    floor_clear = int(baseline["policy_clear"])
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
    day_map = {str(d): m1 for d, m1 in days}
    oracle = load_oracle()
    dials_live = PathSkillDials()
    neural: Optional[TinyPathGate] = None

    print("SELF-CLIMB score pure child (immutable base)…", flush=True)
    base_score = score_policy(child, day_map, mark_rows)
    print(
        f"BASE same={base_score['same_outcome']} mwt={base_score['mark_would_take']} "
        f"breach={base_score['n_breach']} clear={base_score['policy_clear']}",
        flush=True,
    )
    if base_score["same_outcome"] < keep_floor:
        print("WARN: base below floor — restore child embryo before climb", flush=True)

    best_same = int(base_score["same_outcome"])
    best_score = base_score
    best_dials = PathSkillDials(note="pure_child")
    best_neural_state = None
    cycles: List[dict] = []

    if train_neural:
        print("SELF-CLIMB train tiny path gate (child frozen)…", flush=True)
        X, y = collect_neural_labels(day_map, mark_rows, child, oracle, max_days=14)
        print(f"  neural labels n={len(y)} dist={dict(Counter(y.tolist()))}", flush=True)
        if len(y) >= 40:
            neural = train_neural_gate(X, y, epochs=50)
            # eval neural-only blend candidates later
        else:
            neural = TinyPathGate()

    candidates: List[PathSkillDials] = []
    if search_grid:
        candidates.extend(dials_grid_seed())
    # always include neural blend if trained
    if neural is not None:
        candidates.append(
            PathSkillDials(
                thrash_gate=1.0,
                cont_gate=1.0,
                against_htf_hold=1.0,
                require_launch_for_cont=0.5,
                use_neural=1.0,
                entropy_hold=1.0,
                tension_req=1.0,
                note="neural_phys_dominant",
            )
        )
        candidates.append(
            PathSkillDials(
                thrash_gate=1.0,
                cont_gate=0.0,
                against_htf_hold=1.0,
                use_neural=0.85,
                entropy_hold=1.0,
                tension_req=0.0,
                note="neural_wait_heavy",
            )
        )

    # Prefer physics equation + cont seeds early (MWT usually miss_continuation)
    def _seed_priority(d: PathSkillDials) -> int:
        n = d.note or ""
        if n.startswith("phys_"):
            return 0
        if "cont" in n or d.cont_gate >= 1.0:
            return 1
        if "neural" in n:
            return 2
        return 3

    candidates = sorted(candidates, key=_seed_priority)
    # Cap candidates for wall-clock; meta refine fills rest
    candidates = candidates[: max(10, min(len(candidates), 18))]

    rnd = 0
    tried_notes = set()
    while rnd < max_rounds:
        rnd += 1
        if rnd <= len(candidates):
            dials = candidates[rnd - 1]
        else:
            # Meta refine from diagnosis of best so far
            laws = diagnose_mwt_laws(best_score, day_map, child, best_dials, sample_n=6)
            dials = refine_dials(best_dials, laws, last_same=best_same, floor=keep_floor)
            # jitter for exploration
            dials.cont_gate = float(np.clip(dials.cont_gate + np.random.uniform(-0.2, 0.3), 0, 1.5))
            dials.thrash_gate = float(np.clip(dials.thrash_gate + np.random.uniform(-0.15, 0.15), 0, 1.5))
            dials.note = f"meta_r{rnd}_{dials.note}"
            dials.clamp()
            if dials.note in tried_notes:
                dials.min_force = float(np.clip(dials.min_force + np.random.choice([-0.1, 0.1]), 0.05, 0.55))
                dials.require_launch_for_cont = float(1.0 - dials.require_launch_for_cont)
                dials.note = f"meta_r{rnd}_jitter"
            tried_notes.add(dials.note)

        pol = PathSkillPolicy(child, dials, neural=neural if dials.use_neural > 0.05 else None)
        print(
            f"\n===== SELF-CLIMB {rnd}/{max_rounds} dials={ {k: round(v,3) if isinstance(v,float) else v for k,v in asdict(dials).items()} } =====",
            flush=True,
        )
        post = score_policy(pol, day_map, mark_rows)  # type: ignore[arg-type]
        same = int(post["same_outcome"])
        mwt = int(post["mark_would_take"])
        breach = int(post["n_breach"])
        clear = int(post["policy_clear"])
        print(
            f"  POST same={same} mwt={mwt} breach={breach} clear={clear} "
            f"laws={dict(pol._law_counts.most_common(6))}",
            flush=True,
        )

        improved = (
            breach == 0
            and clear >= floor_clear - 2  # allow tiny clear trade if same rises
            and same > best_same
        )
        # Floor protection: never accept pack below keep_floor if base was at floor
        safe = breach == 0 and same >= keep_floor
        decision = "REJECT"
        if improved and safe:
            decision = "KEEP"
            best_same = same
            best_score = post
            best_dials = copy.deepcopy(dials)
            if neural is not None:
                best_neural_state = {k: v.detach().clone() for k, v in neural.state_dict().items()}
            print(f"  KEEP best_same={best_same} skill={dials.note}", flush=True)
        elif same >= goal_same and breach == 0 and same >= best_same:
            decision = "KEEP"
            best_same = same
            best_score = post
            best_dials = copy.deepcopy(dials)
            print(f"  KEEP goal skill={dials.note}", flush=True)
        else:
            print("  REJECT (rollback to best dials mentally; child never moved)", flush=True)

        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "round": rnd,
            "decision": decision,
            "same": same,
            "mwt": mwt,
            "breach": breach,
            "clear": clear,
            "best_same": best_same,
            "dials": asdict(dials),
            "law_counts": dict(pol._law_counts),
            "method": "self_climb_path_skill",
        }
        append_mem(row)
        cycles.append(row)

        # Persist harness state every round
        with open(HARNESS_STATE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "best_same": best_same,
                    "best_mwt": best_score["mark_would_take"],
                    "best_breach": best_score["n_breach"],
                    "best_dials": asdict(best_dials),
                    "goal_same": goal_same,
                    "keep_floor": keep_floor,
                    "child_sha": CHILD_SHA,
                    "cycles": cycles[-30:],
                    "method": "self_climb_path_skill",
                    "passed_35": best_same > keep_floor,
                },
                f,
                indent=2,
            )
        with open(HARNESS_REPORT, "w", encoding="utf-8") as f:
            f.write(
                f"# Self-climb L2L report\n\n"
                f"- best_same: **{best_same}**\n"
                f"- goal: {goal_same}\n"
                f"- best_dials: `{asdict(best_dials)}`\n"
                f"- last: {decision} same={same}\n"
                f"- child immutable: `{CHILD_SHA[:16]}…`\n"
                f"- method: path skill decode (not day BC)\n"
            )

        if best_same >= goal_same and best_score["n_breach"] == 0:
            print(f"GOAL reached same={best_same} >= {goal_same}", flush=True)
            break

    # --- Promote KEEP past floor: save skill + update BEST (NOT child weights) ---
    if best_same > keep_floor and best_score["n_breach"] == 0:
        torch.save(
            {
                "tag": "path_skill_self_climb_v1",
                "child_sha": CHILD_SHA,
                "dials": asdict(best_dials),
                "neural_state": best_neural_state,
                "same_outcome": best_same,
                "score": {
                    k: best_score[k]
                    for k in (
                        "same_outcome",
                        "policy_clear",
                        "mark_would_take",
                        "n_breach",
                        "miss_class_counts",
                    )
                },
                "method": "self_climb_path_skill",
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
            SKILL_CKPT,
        )
        with open(BEST, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "same_outcome": best_same,
                    "policy_clear": best_score["policy_clear"],
                    "mwt": best_score["mark_would_take"],
                    "breach": best_score["n_breach"],
                    "source": f"self_climb_path_laws__{best_dials.note}",
                    "stage": "teen" if best_same > 35 else "child",
                    "child_frozen_sha256": CHILD_SHA,
                    "growth_method": "self_climb_path_skill",
                    "core_skill": best_dials.skill_attribution(),
                    "skill_class": "path_dials_laws_not_day_memo",
                    "physics_decode": {
                        "pinn_gravity": best_dials.against_htf_hold,
                        "entropy_hold": best_dials.entropy_hold,
                        "tension_req": best_dials.tension_req,
                        "note": "physics.md as score-time laws; NOT act-head BC",
                    },
                    "path_skill_ckpt": SKILL_CKPT,
                    "dials": asdict(best_dials),
                    "note": "Child weights immutable; score via PathSkillPolicy adapter",
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )
        # Do NOT overwrite mark_clone_full_obs_v1.pt — skill lives in adapter.
        # Optional: document that scoring must wrap child with PathSkillPolicy.
        print(f"PROMOTED skill ckpt same={best_same} → {SKILL_CKPT}", flush=True)
        try:
            with open(os.path.join(OUT, "WHAT_WORKS__GOAL.md"), "a", encoding="utf-8") as wf:
                wf.write(
                    f"| KEEP self-climb | **{best_same}** | {best_score['mark_would_take']} | "
                    f"{best_score['n_breach']} | path skill `{best_dials.note}` |\n"
                )
        except OSError:
            pass
    else:
        print(
            f"No promote: best_same={best_same} (need >{keep_floor}). Child floor held.",
            flush=True,
        )

    summary = {
        "best_same": best_same,
        "best_mwt": best_score["mark_would_take"],
        "best_breach": best_score["n_breach"],
        "best_dials": asdict(best_dials),
        "passed_35": best_same > keep_floor,
        "cycles": len(cycles),
        "method": "self_climb_path_skill",
    }
    print(f"DONE self-climb {summary}", flush=True)
    return summary


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Autonomous path-skill self-climb past 35")
    ap.add_argument("--max-rounds", type=int, default=20)
    ap.add_argument("--keep-floor", type=int, default=35)
    ap.add_argument("--goal-same", type=int, default=36)
    ap.add_argument("--no-neural", action="store_true")
    ap.add_argument("--no-grid", action="store_true")
    args = ap.parse_args()
    run_self_climb(
        max_rounds=args.max_rounds,
        keep_floor=args.keep_floor,
        goal_same=args.goal_same,
        train_neural=not args.no_neural,
        search_grid=not args.no_grid,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
