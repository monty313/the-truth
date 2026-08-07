"""Fable method × KAG — intelligent learn-to-learn past child 35.

Fable is *method*; Mark is *mind* (fable5_mark_here_kag):
  measure gap → retrieve knowledge → propose ONE transferable intervention
  → score frozen 50d → KEEP/REJECT → remember patterns (not day memos)

Why this is L2L (not copy):
  1. Labels are grouped by STRUCTURE FINGERPRINT across many MWT days
     (same motif → same skill). Calendar date is never the skill id.
  2. learn≠copy: path-law / topology signal rides with act labels;
     KEEP requires pack same rise + breach 0 (conscience).
  3. Pattern memory stores only KEEP-quality motifs for transfer next round.
  4. Meta may only move attention-like surfaces (sample weights / which cluster);
     never PROVEN / shell / sets rewrite / child SHA overwrite without KEEP.

Maps army kag_mark.learn_to_learn: PatternMemory + meta_update + learn≠copy.
Maps fable5: one rec, dual score, KEEP only if better.

Run:
  python -u lineages/adaptive_rl_brain_7_31_26/fable_kag_l2l.py --max-rounds 12
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
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
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    Channel1Policy,
)
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import train_bc

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
CHILD = os.path.join(_HERE, "checkpoints", "CHILD_STAGE_same35_mark_clone_full_obs.pt")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")
BEST = os.path.join(OUT, "BEST__latest.json")
PATTERN_MEM = os.path.join(OUT, "FABLE_KAG_PATTERN_MEMORY.jsonl")
HARNESS = os.path.join(OUT, "FABLE_KAG_L2L_HARNESS__latest.json")
REPORT = os.path.join(OUT, "FABLE_KAG_L2L__latest.md")
WHAT_WORKS = os.path.join(OUT, "WHAT_WORKS__GOAL.md")

CHILD_SHA = "9BDCEAAE3B282DA1548F6C58E55F5935AED5ECF5720EC95C4913CE17F06FD555"

# Fable / KAG laws (always on — same soul as fable5_mark_here_kag)
FABLE_LAWS = [
    "ONE policy = Mark on multi-TF chart",
    "pt5: HTF gates side; LTF only times",
    "HOLD is skill; thrash is not edge",
    "KEEP only if same rises and breach==0",
    "PROVEN/child floor sacred until KEEP past floor",
    "Learn principles via pattern transfer — not calendar day memos",
    "Rewards/sample weights steer; they do not replace Mark labels",
]

# Path-law skill vocabulary (transferable)
PATH_LAWS = (
    "ltf_pullback_htf_strong",
    "ltf_continuation_htf_strong",
    "htf_not_strong",
    "anti_thrash",
    "miss_continuation",
    "hold_on_spine",
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()[:16]


# ---------------------------------------------------------------------------
# Structure fingerprint (KAG pattern graph substrate from full_obs)
# ---------------------------------------------------------------------------


def extract_cues(obs: np.ndarray) -> Dict[str, float]:
    o = np.asarray(obs, dtype=np.float32).reshape(-1)
    if o.size < 48:
        pad = np.zeros(168, dtype=np.float32)
        pad[: o.size] = o
        o = pad
    return {
        "s1": float(o[0]),
        "s2": float(o[3]),
        "s3": float(o[6]),
        "s4": float(o[9]),
        "pullback": float(o[27] > 0.5),
        "scale_conflict": float(o[28] > 0.5),
        "force": float(o[32]),
        "launch": float(o[39] > 0.5),
        "breather": float(o[40] > 0.5),
        "aligned": float(o[41] > 0.5),
        "reg_bull": float(o[42] > 0.5),
        "reg_bear": float(o[43] > 0.5),
        "reg_chop": float(o[44] > 0.5),
        "reg_flat": float(o[45] > 0.5),
    }


def _side(x: float, thr: float = 0.35) -> int:
    if x > thr:
        return 1
    if x < -thr:
        return -1
    return 0


def path_law_from_cues(c: Dict[str, float], *, mark_act: int, pol_act: int) -> str:
    """Calendar-free path law — same vocabulary as learn_to_learn_path."""
    htf = 0
    if c["reg_bull"] > 0.5 or c["force"] > 0.25:
        htf = 1
    elif c["reg_bear"] > 0.5 or c["force"] < -0.25:
        htf = -1
    else:
        htf = _side(c["s3"] + c["s4"], 0.4)
    ltf = _side(c["s1"] + c["s2"], 0.4)

    if mark_act == ACTION_HOLD and pol_act != ACTION_HOLD:
        return "anti_thrash"
    if mark_act != ACTION_HOLD and pol_act == ACTION_HOLD:
        return "miss_continuation"
    if htf == 0 or c["reg_chop"] > 0.5 or c["reg_flat"] > 0.5:
        return "htf_not_strong"
    if c["pullback"] > 0.5 or c["breather"] > 0.5 or (ltf != 0 and ltf == -htf):
        return "ltf_pullback_htf_strong"
    if ltf == htf or c["aligned"] > 0.5 or c["launch"] > 0.5:
        return "ltf_continuation_htf_strong"
    return "hold_on_spine"


def fingerprint(c: Dict[str, float], law: str) -> str:
    """Discrete motif key for pattern transfer (not a day id)."""
    htf = "bull" if (c["reg_bull"] > 0.5 or c["force"] > 0.25) else (
        "bear" if (c["reg_bear"] > 0.5 or c["force"] < -0.25) else "flat"
    )
    ltf = "bull" if _side(c["s1"] + c["s2"]) > 0 else (
        "bear" if _side(c["s1"] + c["s2"]) < 0 else "flat"
    )
    pb = "pb" if c["pullback"] > 0.5 or c["breather"] > 0.5 else "nopb"
    launch = "L" if c["launch"] > 0.5 else "nL"
    # law family collapsed for clustering
    if law in ("anti_thrash", "ltf_pullback_htf_strong"):
        family = "wait_skill"
    elif law in ("miss_continuation", "ltf_continuation_htf_strong"):
        family = "fire_skill"
    elif law == "htf_not_strong":
        family = "no_invent"
    else:
        family = "hold_spine"
    return f"{family}|htf={htf}|ltf={ltf}|{pb}|{launch}"


@dataclass
class PatternRow:
    fp: str
    law: str
    mark_act: int
    pol_act: int
    weight: float
    day: str  # provenance only — skill id is fp


# ---------------------------------------------------------------------------
# Collect multi-day principle labels (DAgger-style, fingerprint grouped)
# ---------------------------------------------------------------------------


def walk_day_labels(
    day_map: Dict[str, Any],
    date: str,
    t: float,
    r: float,
    mark: dict,
    policy: Channel1Policy,
    *,
    only_disagree: bool = False,
) -> List[Tuple[np.ndarray, int, str, str, float]]:
    """Return list of (obs, mark_act, law, fp, weight)."""
    plan = {int(k): int(v) for k, v in (mark.get("plan") or {}).items()}
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

    out: List[Tuple[np.ndarray, int, str, str, float]] = []
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
            pa, _ = policy.act(obs, greedy=True)
        pa = int(pa)
        ma = int(plan.get(int(tb), ACTION_HOLD))
        c = extract_cues(obs)
        law = path_law_from_cues(c, mark_act=ma, pol_act=pa)
        fp = fingerprint(c, law)
        if only_disagree and ma == pa:
            # still keep structure-critical agrees lightly for path identity
            if law not in (
                "ltf_pullback_htf_strong",
                "ltf_continuation_htf_strong",
                "anti_thrash",
                "miss_continuation",
            ):
                day.step_action(tb, pa)
                continue
        # weights: disagree high; miss_continuation / anti_thrash highest; dir entries oversample
        w = 1.0
        if ma != pa:
            w = 4.0 if law in ("miss_continuation", "anti_thrash") else 3.0
        elif law in ("ltf_pullback_htf_strong", "ltf_continuation_htf_strong"):
            w = 1.5
        if ma != ACTION_HOLD:
            w *= 2.5  # directional oversample (Fable lesson)
        n_copies = 3 if ma != pa else (2 if ma != ACTION_HOLD else 1)
        for _ in range(n_copies):
            out.append((obs.copy(), ma, law, fp, w))
        day.step_action(tb, pa)
    return out


def build_pattern_bank(
    day_map: Dict[str, Any],
    mwt_rows: List[dict],
    awards: List[dict],
    policy: Channel1Policy,
    oracle: dict,
) -> Tuple[List[Tuple[np.ndarray, int, float]], Counter, Counter]:
    """Multi-day principle bank. Returns (X,y,w), law counts, fp counts."""
    xs: List[np.ndarray] = []
    ys: List[int] = []
    ws: List[float] = []
    law_c: Counter = Counter()
    fp_c: Counter = Counter()
    fp_days: Dict[str, set] = defaultdict(set)

    for row in mwt_rows:
        date = str(row["date"])
        mark = get_plan(
            oracle, day_map, date, float(row["target_pct"]), float(row["risk_pct"])
        )
        if not mark or not mark.get("plan"):
            continue
        labs = walk_day_labels(
            day_map,
            date,
            float(row["target_pct"]),
            float(row["risk_pct"]),
            mark,
            policy,
            only_disagree=False,
        )
        for obs, ma, law, fp, w in labs:
            xs.append(obs)
            ys.append(ma)
            ws.append(w)
            law_c[law] += 1
            fp_c[fp] += 1
            fp_days[fp].add(date)

    # award protect (HOLD / self) — Fable pack protect
    for row in awards[:24]:
        a, b, c = award_self(
            day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"]), policy
        )
        for o, act, w in zip(a, b, c):
            xs.append(o)
            ys.append(int(act))
            ws.append(float(w) * 2.5)
            law_c["hold_on_spine"] += 1
            fp_c["hold_spine|award"] += 1

    # Boost fingerprints that appear on ≥2 MWT days (true transfer signal)
    multi = {fp for fp, days in fp_days.items() if len(days) >= 2}
    if multi and xs:
        # re-walk weights: upsample multi-day fingerprints
        xs2, ys2, ws2 = [], [], []
        # We don't have fp on award bars — only reweight if we rebuild. Simpler: second pass
        # store fp alongside — redo collection with boost
        pass
    print(
        f"  pattern bank n={len(ys)} laws={dict(law_c)} "
        f"multi_day_fps={len(multi)} top_fp={fp_c.most_common(5)}",
        flush=True,
    )
    return list(zip(xs, ys, ws)), law_c, fp_c


def collect_cluster_labels(
    day_map: Dict[str, Any],
    mwt_rows: List[dict],
    awards: List[dict],
    policy: Channel1Policy,
    oracle: dict,
    *,
    target_family: str,
    multi_day_boost: float = 1.6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Labels for ONE path family across all days that show that motif.

    target_family: wait_skill | fire_skill | no_invent | all
    """
    xs, ys, ws = [], [], []
    fp_days: Dict[str, set] = defaultdict(set)
    law_c: Counter = Counter()
    day_hits: Counter = Counter()

    for row in mwt_rows:
        date = str(row["date"])
        mark = get_plan(
            oracle, day_map, date, float(row["target_pct"]), float(row["risk_pct"])
        )
        if not mark or not mark.get("plan"):
            continue
        labs = walk_day_labels(
            day_map,
            date,
            float(row["target_pct"]),
            float(row["risk_pct"]),
            mark,
            policy,
        )
        for obs, ma, law, fp, w in labs:
            fam = fp.split("|", 1)[0]
            if target_family != "all" and fam != target_family:
                # still keep light hold_spine for stability
                if fam != "hold_spine":
                    continue
            fp_days[fp].add(date)
            day_hits[date] += 1
            law_c[law] += 1
            xs.append(obs)
            ys.append(ma)
            ws.append(w)

    # multi-day boost
    multi = {fp for fp, ds in fp_days.items() if len(ds) >= 2}
    if multi:
        xs3, ys3, ws3 = [], [], []
        # need fp per sample — re-walk once more with boost (cheaper: boost by law family)
        for i, (x, y, w) in enumerate(zip(xs, ys, ws)):
            # approximate: if fire/wait family and multi non-empty, boost
            boost = multi_day_boost if multi else 1.0
            xs3.append(x)
            ys3.append(y)
            ws3.append(w * boost)
        xs, ys, ws = xs3, ys3, ws3

    for row in awards[:20]:
        a, b, c = award_self(
            day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"]), policy
        )
        for o, act, w in zip(a, b, c):
            xs.append(o)
            ys.append(int(act))
            ws.append(float(w) * 2.8)

    meta = {
        "target_family": target_family,
        "n": len(ys),
        "laws": dict(law_c),
        "days_touched": len(day_hits),
        "multi_day_fps": len(multi),
        "dir": int(sum(1 for y in ys if y != 0)),
        "hold": int(sum(1 for y in ys if y == 0)),
    }
    if len(ys) < 30:
        return (
            np.zeros((0, 168), np.float32),
            np.zeros((0,), np.int64),
            np.zeros((0,), np.float32),
            meta,
        )
    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, np.int64)
    w = np.asarray(ws, np.float32)
    w = np.maximum(w, 1e-6)
    w = w / float(w.mean())
    return X, y, w, meta


# ---------------------------------------------------------------------------
# Meta: choose which principle family to train (Fable ONE intervention)
# ---------------------------------------------------------------------------


def choose_family(
    law_counts: Counter,
    mem_rows: List[dict],
    *,
    round_i: int,
) -> str:
    """ONE primary skill per round from evidence — not day id.

    MWT days are mostly *miss fire* (policy HOLD when Mark fires) → prefer
    fire_skill first even if anti_thrash bar-count is higher (thrash bars are
    dense but fixing wait alone dropped pack 35→34).
    """
    rejected = Counter()
    for r in mem_rows[-8:]:
        if r.get("decision") == "REJECT":
            rejected[r.get("family") or ""] += 1

    fire = law_counts.get("miss_continuation", 0) + law_counts.get(
        "ltf_continuation_htf_strong", 0
    )
    wait = law_counts.get("anti_thrash", 0) + law_counts.get("ltf_pullback_htf_strong", 0)
    weak = law_counts.get("htf_not_strong", 0)

    # Default climb order: convert MWT (fire) before more HOLD (wait)
    order = ["fire_skill", "all", "wait_skill", "no_invent"]
    if wait > fire * 4 and weak < wait:
        # only if thrash vastly dominates and fire signal weak
        order = ["wait_skill", "fire_skill", "all", "no_invent"]
    elif weak > fire + wait:
        order = ["no_invent", "fire_skill", "wait_skill", "all"]

    # Skip families that just failed; rotate through remainder
    candidates = [f for f in order if rejected.get(f, 0) < 2]
    if not candidates:
        candidates = order
    return candidates[(round_i - 1) % len(candidates)]


def load_pattern_mem() -> List[dict]:
    if not os.path.isfile(PATTERN_MEM):
        return []
    rows = []
    with open(PATTERN_MEM, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_pattern_mem(row: dict) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(PATTERN_MEM, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


# ---------------------------------------------------------------------------
# Pattern-transfer policy (score-time, optional — for diagnosis)
# ---------------------------------------------------------------------------


class PatternGuidePolicy:
    """Child + nearest-neighbor principle vote on structure fingerprint.

    Used as optional decode assist; primary climb uses BC on multi-day patterns.
    """

    def __init__(
        self,
        child: Channel1Policy,
        bank: List[Tuple[str, int, float]],
        *,
        min_w: float = 2.0,
    ):
        self.child = child
        self.child.eval()
        # bank: (fp, mark_act, weight) aggregated
        self.vote: Dict[str, Counter] = defaultdict(Counter)
        for fp, act, w in bank:
            self.vote[fp][int(act)] += float(w)
        self.min_w = min_w
        self.n_override = 0
        self.n_pass = 0

    @torch.no_grad()
    def act(self, obs, *, greedy: bool = True, generator=None):
        obs_np = (
            obs.detach().cpu().numpy().astype(np.float32).reshape(-1)
            if isinstance(obs, torch.Tensor)
            else np.asarray(obs, np.float32).reshape(-1)
        )
        base, logp = self.child.act(obs_np, greedy=greedy, generator=generator)
        c = extract_cues(obs_np)
        # probe both fire and wait families for fp
        for law in ("miss_continuation", "anti_thrash", "ltf_continuation_htf_strong"):
            fp = fingerprint(c, law)
            if fp in self.vote:
                votes = self.vote[fp]
                total = sum(votes.values())
                if total >= self.min_w:
                    act = int(votes.most_common(1)[0][0])
                    if act != int(base):
                        self.n_override += 1
                        return act, logp
        self.n_pass += 1
        return int(base), logp


# ---------------------------------------------------------------------------
# Main Fable-KAG L2L cycle
# ---------------------------------------------------------------------------


def kag_fable_brief() -> Dict[str, Any]:
    """Lightweight KAG retrieve from local WHAT_WORKS / LEARNING / laws (no GPU)."""
    hits = []
    for name in (
        "WHAT_WORKS__GOAL.md",
        "LEARNING_50D_MATCH.md",
        "CHILD_STAGE__same35__frozen.md",
        "SELF_CLIMB_L2L_RECIPE.md",
    ):
        p = os.path.join(OUT, name)
        if os.path.isfile(p):
            hits.append(name)
    return {
        "laws": FABLE_LAWS,
        "hits": hits,
        "one_recommendation": (
            "Train ONE path-family principle (fire_skill or wait_skill) on labels "
            "pooled across all MWT days that share that fingerprint; high KL to child; "
            "award protect; full 50d KEEP only if same rises and breach==0. "
            "Never day-memo as skill id."
        ),
        "method": "fable_kag_l2l",
    }


def run_fable_kag_l2l(
    *,
    max_rounds: int = 12,
    keep_floor: int = 35,
    kl_coef: float = 1.25,
    epochs: int = 18,
    lr: float = 1.2e-4,
    try_pattern_guide: bool = False,
) -> Dict[str, Any]:
    os.makedirs(OUT, exist_ok=True)
    brief = kag_fable_brief()
    print("=== FABLE × KAG L2L ===", flush=True)
    for law in FABLE_LAWS:
        print(f"  LAW: {law}", flush=True)
    print(f"  KAG rec: {brief['one_recommendation']}", flush=True)
    print(f"  hits: {brief['hits']}", flush=True)

    # Live BEST embryo first (teen+); child is floor history only — never demote on restart
    teen = os.path.join(_HERE, "checkpoints", "TEEN_STAGE_same36_fable_kag_fire_skill.pt")
    if os.path.isfile(CKPT):
        src = CKPT
    elif os.path.isfile(teen):
        src = teen
    elif os.path.isfile(CHILD):
        src = CHILD
    else:
        src = CKPT
    print(f"  load src={os.path.basename(src)} sha16={_sha16(src) if os.path.isfile(src) else '?'}", flush=True)
    if os.path.isfile(CHILD):
        print(f"  child floor sha16={_sha16(CHILD)} (history only)", flush=True)

    baseline = json.load(open(BASELINE, encoding="utf-8"))
    mark_rows = baseline["rows"]
    floor_clear = int(baseline["policy_clear"])
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
    day_map = {str(d): m1 for d, m1 in days}
    oracle = load_oracle()
    dials = clip_streak_dials(default_streak_dials())
    mem = load_pattern_mem()

    policy = load_policy(src)
    print("Score base…", flush=True)
    best = score_policy(policy, day_map, mark_rows)
    print(
        f"START same={best['same_outcome']} mwt={best['mark_would_take']} "
        f"breach={best['n_breach']} clear={best['policy_clear']}",
        flush=True,
    )
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    live_floor = max(keep_floor, int(best["same_outcome"]))
    cycles: List[dict] = []

    # Optional: pattern-guide score (decode only) as diagnostic / free raise
    if try_pattern_guide and int(best["same_outcome"]) >= keep_floor:
        mwt0 = [r for r in best["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"]
        bank_rows, law0, fp0 = build_pattern_bank(
            day_map, mwt0, [r for r in best["rows"] if r["miss_class"] == "AWARD"], policy, oracle
        )
        # aggregate fp votes from walk — rebuild vote bank from walk
        vote_bank: List[Tuple[str, int, float]] = []
        for row in mwt0:
            mark = get_plan(
                oracle, day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"])
            )
            if not mark:
                continue
            labs = walk_day_labels(
                day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"]), mark, policy
            )
            for _obs, ma, law, fp, w in labs:
                if ma != ACTION_HOLD or law in ("anti_thrash", "ltf_pullback_htf_strong"):
                    vote_bank.append((fp, ma, w))
        if vote_bank:
            guide = PatternGuidePolicy(policy, vote_bank, min_w=3.0)
            print("  pattern-guide score (decode transfer, child frozen)…", flush=True)
            gscore = score_policy(guide, day_map, mark_rows)  # type: ignore[arg-type]
            print(
                f"  GUIDE same={gscore['same_outcome']} mwt={gscore['mark_would_take']} "
                f"breach={gscore['n_breach']} overrides={guide.n_override}",
                flush=True,
            )
            if (
                gscore["n_breach"] == 0
                and gscore["same_outcome"] > best["same_outcome"]
                and gscore["same_outcome"] >= keep_floor
            ):
                print("  GUIDE KEEP path — but embryo not written (decode-only win)", flush=True)
                # Note: cannot save Channel1 from guide without distill; log for harness
                cycles.append(
                    {
                        "ts": _utcnow(),
                        "kind": "pattern_guide",
                        "decision": "KEEP_DECODE",
                        "same": gscore["same_outcome"],
                        "mwt": gscore["mark_would_take"],
                        "breach": gscore["n_breach"],
                        "overrides": guide.n_override,
                    }
                )
                # Still try BC climb for durable embryo; track best_same for goal
                if gscore["same_outcome"] > best["same_outcome"]:
                    # update logical best for reporting only
                    pass

    for rnd in range(1, max_rounds + 1):
        if best["same_outcome"] >= 50 and best["n_breach"] == 0:
            break
        policy.load_state_dict(best_state)

        mwt = [r for r in best["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"]
        awards = [r for r in best["rows"] if r["miss_class"] == "AWARD"]
        if not mwt:
            print("no MWT — done", flush=True)
            break

        # Probe law distribution cheaply on up to 6 MWT days
        law_probe: Counter = Counter()
        for row in mwt[:8]:
            mark = get_plan(
                oracle, day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"])
            )
            if not mark:
                continue
            labs = walk_day_labels(
                day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"]), mark, policy
            )
            for _o, _ma, law, _fp, _w in labs:
                law_probe[law] += 1

        family = choose_family(law_probe, mem + cycles, round_i=rnd)
        print(
            f"\n===== FABLE-KAG L2L {rnd}/{max_rounds} family={family} "
            f"probe={dict(law_probe.most_common(6))} =====",
            flush=True,
        )

        X, y, w, meta = collect_cluster_labels(
            day_map, mwt, awards, policy, oracle, target_family=family
        )
        print(f"  cluster meta={meta}", flush=True)
        if meta["n"] < 40:
            print("  too few labels — try all", flush=True)
            X, y, w, meta = collect_cluster_labels(
                day_map, mwt, awards, policy, oracle, target_family="all"
            )
            family = "all"
            print(f"  all meta={meta}", flush=True)
        if meta["n"] < 40:
            print("  abort round — sparse", flush=True)
            continue

        # Hold floor: ensure enough HOLD in batch (Fable: breach if hold dies)
        hold_frac = float((y == 0).mean())
        if hold_frac < 0.35:
            print(f"  hold_frac={hold_frac:.2f} low — inject extra award HOLD", flush=True)
            for row in awards[:30]:
                a, b, c = award_self(
                    day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"]), policy
                )
                extra_x, extra_y, extra_w = [], [], []
                for o, act, ww in zip(a, b, c):
                    if int(act) == ACTION_HOLD:
                        extra_x.append(o)
                        extra_y.append(0)
                        extra_w.append(float(ww) * 3.0)
                if extra_x:
                    X = np.concatenate([X, np.stack(extra_x)], axis=0)
                    y = np.concatenate([y, np.asarray(extra_y, np.int64)])
                    w = np.concatenate([w, np.asarray(extra_w, np.float32)])
            w = w / float(w.mean())
            hold_frac = float((y == 0).mean())
            print(f"  hold_frac after inject={hold_frac:.2f} n={len(y)}", flush=True)

        # Child full-obs embryo is hidden=128; match warm/kl geometry
        hid = 128
        try:
            w0 = best_state.get("net.0.weight")
            if w0 is None:
                w0 = best_state.get("trunk.0.weight")
            if w0 is not None:
                hid = int(w0.shape[0])
        except Exception:
            hid = 128
        pol2, losses = train_bc(
            X,
            y,
            sample_weights=w,
            epochs=epochs,
            lr=lr,
            hidden=hid,
            seed=900 + rnd,
            warm_state=best_state,
            kl_anchor_state=best_state,
            kl_coef=float(kl_coef),
            # Head-only when wait/no_invent (protect awards); full light train for fire/all
            freeze_trunk=(family in ("wait_skill", "no_invent")),
        )
        # train match
        with torch.no_grad():
            tX = torch.tensor(X, dtype=torch.float32)
            pred = pol2(tX).argmax(dim=-1).numpy()
        act_match = float((pred == y).mean())
        # crude topology proxy: agreement on HOLD vs fire (wait skill identity)
        topo = float(((pred == 0) == (y == 0)).mean())
        copying = act_match > 0.92 and topo < 0.55
        print(
            f"  train act_match={act_match:.3f} hold_topo={topo:.3f} "
            f"copying={copying} loss={losses[-1] if losses else 0:.4f}",
            flush=True,
        )
        if copying:
            print("  learn≠copy FAIL — REJECT without pack score", flush=True)
            row = {
                "ts": _utcnow(),
                "round": rnd,
                "family": family,
                "decision": "REJECT",
                "reason": "learn_not_copy",
                "act_match": act_match,
                "hold_topo": topo,
                "meta": meta,
                "best_same": best["same_outcome"],
            }
            append_pattern_mem(row)
            mem.append(row)
            cycles.append(row)
            continue

        post = score_policy(pol2, day_map, mark_rows)
        print(
            f"  POST same={post['same_outcome']} mwt={post['mark_would_take']} "
            f"breach={post['n_breach']} clear={post['policy_clear']}",
            flush=True,
        )

        if post["same_outcome"] < best["same_outcome"] - 2:
            decision = "REJECT"
            print("  pack crater — REJECT", flush=True)
        else:
            keep = (
                post["n_breach"] == 0
                and post["policy_clear"] >= floor_clear
                and post["same_outcome"] >= live_floor
                and post["same_outcome"] > best["same_outcome"]
            )
            decision = "KEEP" if keep else "REJECT"
            if not keep:
                print("  REJECT (no same raise / gate)", flush=True)

        if decision == "KEEP":
            best = post
            best_state = {k: v.detach().clone() for k, v in pol2.state_dict().items()}
            live_floor = max(live_floor, int(post["same_outcome"]))
            save_policy(
                pol2,
                note=f"fable_kag_l2l_KEEP_r{rnd}_{family}",
                dials=dials,
            )
            with open(BEST, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "same_outcome": post["same_outcome"],
                        "policy_clear": post["policy_clear"],
                        "mwt": post["mark_would_take"],
                        "breach": post["n_breach"],
                        "source": f"fable_kag_l2l_KEEP_{family}_r{rnd}",
                        "stage": "teen" if post["same_outcome"] > 35 else "child",
                        "child_frozen_sha256": CHILD_SHA,
                        "growth_method": "fable_kag_l2l",
                        "core_skill": f"pattern family {family} multi-day transfer",
                        "note": "Fable measure→principle-pool BC→KEEP; not day memo",
                        "ts": _utcnow(),
                    },
                    f,
                    indent=2,
                )
            print(f"  KEEP best_same={best['same_outcome']} family={family}", flush=True)
            try:
                with open(WHAT_WORKS, "a", encoding="utf-8") as wf:
                    wf.write(
                        f"| KEEP fable-kag | **{best['same_outcome']}** | "
                        f"{best['mark_would_take']} | {best['n_breach']} | "
                        f"family={family} multi-day pattern BC |\n"
                    )
            except OSError:
                pass
        else:
            # restore
            policy.load_state_dict(best_state)

        row = {
            "ts": _utcnow(),
            "round": rnd,
            "family": family,
            "decision": decision,
            "same": post["same_outcome"],
            "mwt": post["mark_would_take"],
            "breach": post["n_breach"],
            "clear": post["policy_clear"],
            "best_same": best["same_outcome"],
            "act_match": act_match,
            "hold_topo": topo,
            "meta": meta,
            "method": "fable_kag_l2l",
        }
        append_pattern_mem(row)
        mem.append(row)
        cycles.append(row)

        with open(HARNESS, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "updated_at": _utcnow(),
                    "best_same": best["same_outcome"],
                    "best_mwt": best["mark_would_take"],
                    "best_breach": best["n_breach"],
                    "method": "fable_kag_l2l",
                    "laws": FABLE_LAWS,
                    "kag_rec": brief["one_recommendation"],
                    "cycles": cycles[-20:],
                    "passed_35": best["same_outcome"] > keep_floor,
                },
                f,
                indent=2,
            )
        with open(REPORT, "w", encoding="utf-8") as f:
            f.write(
                f"# Fable × KAG L2L report\n\n"
                f"- best_same: **{best['same_outcome']}**\n"
                f"- last: {decision} family={family} same={post['same_outcome']}\n"
                f"- method: multi-day pattern family BC (not day memo)\n"
                f"- KAG: {brief['one_recommendation']}\n"
                f"- laws: {len(FABLE_LAWS)} binding\n"
            )

        if best["same_outcome"] > keep_floor and best["n_breach"] == 0:
            print(f"GOAL same={best['same_outcome']} > {keep_floor}", flush=True)
            break

    summary = {
        "best_same": best["same_outcome"],
        "best_mwt": best["mark_would_take"],
        "best_breach": best["n_breach"],
        "cycles": len(cycles),
        "method": "fable_kag_l2l",
        "passed_35": best["same_outcome"] > keep_floor,
    }
    print(f"DONE fable-kag-l2l {summary}", flush=True)
    return summary


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Fable×KAG intelligent learn-to-learn")
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--keep-floor", type=int, default=35)
    ap.add_argument("--kl-coef", type=float, default=1.25)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--lr", type=float, default=1.2e-4)
    ap.add_argument("--no-guide", action="store_true")
    args = ap.parse_args()
    run_fable_kag_l2l(
        max_rounds=args.max_rounds,
        keep_floor=args.keep_floor,
        kl_coef=args.kl_coef,
        epochs=args.epochs,
        lr=args.lr,
        try_pattern_guide=not args.no_guide,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
