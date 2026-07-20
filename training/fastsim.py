"""FastSim — the batched market twin for the GPU Edition (Bot 1.5).

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty (Bot 1.5 GPU Edition, 2026-07-20).
WHAT:  A vectorized (torch) re-implementation of backtesting/simulator.DaySim
       that steps N markets AT ONCE on the GPU. Same physics: ATR-anchored
       stops, spread, gap-aware broker stops, the intrabar worst-case floor
       LAW, the ratchet, the forever-masks as hard law, the heat guard, the
       per-trade cap, the trade budget. Same reward doctrine (training/rewards
       + configs/rewards.yaml), batched. The brain (training/policy.Brain) is
       UNCHANGED and unshared-of-shape — obs is assembled identically to
       training/env.TradingEnv so a PROVEN_2x checkpoint plugs straight in.
WHY:   To run 8,000 instances fast on an L4, the market must be batched math.
       DaySim is built for accuracy, not batch speed. FastSim is the twin;
       the REAL DaySim stays the judge (scripts/gpu_validate.py proves they
       match, and every streak/proof is scored on DaySim, never the twin).
FIDELITY NOTES (honest):
   * Stacks are fixed K slots holding AGGREGATE (units, avg_price, adds) — this
     is EXACT for DaySim, which only ever uses stack.units / stack.avg_price /
     add-count (never the individual entries). Slot exhaustion (all K used)
     rejects further opens, like a limit.
   * Shell re-validation is applied at FILL (the binding moment in DaySim).
   * Anti-gravity training-wheel (decays to ~0) is omitted; idleness hunger is
     approximated as (flat & op==hold). Both are tiny weights. Everything that
     moves P&L or the floor law is faithful; gpu_validate measures the gap.
WHEN:  2026-07-20.
INTERCONNECTED WITH: training/policy.Brain, training/gpu_data, core.configs
       (shell + goals + rewards — the SAME numbers), scripts/gpu_train,
       scripts/gpu_validate.
----------------------------------------------------------------------

CHANGE LOG (newest first — APPEND on every edit with date + WHY; keep this line):
- 2026-07-20  ratchet lock line = goal + flat_cost (was bare goal) — WHY: lift demo
  proved the stand-down flatten banked goal - cost (~2.9% on a 3.0% target); R3#7's
  documented intent is "flatten still realizes >= goal". Mirrored in backtesting/simulator.
- 2026-07-20  created — WHY: 8,000-at-once GPU training needs a batched twin of
  DaySim; brain shape untouched, DaySim stays the judge.
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
from __future__ import annotations
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.configs import shell_cfg as _shell, goals_cfg as _goals, load as _load  # noqa: E402

SELF_DIM = 12
FRAME = 10
NEG = -1.0e9


class FastSim:
    """Batched twin of DaySim. One .step() advances every env one M1 bar."""

    def __init__(self, days_obs, days_phys, day_lens, cols, device="cpu",
                 K: int = 48, eq0: float = 100_000.0):
        self.dev = torch.device(device)
        self.days_obs = torch.as_tensor(days_obs, dtype=torch.float32, device=self.dev)
        self.days_phys = torch.as_tensor(days_phys, dtype=torch.float32, device=self.dev)
        self.day_lens = torch.as_tensor(day_lens, dtype=torch.long, device=self.dev)
        self.Lmax = int(self.days_obs.shape[1])
        self.C = int(self.days_obs.shape[2])
        self.K = K
        self.eq0 = float(eq0)

        # ----- physics constants: the SAME single-door numbers -----
        sc = _shell()
        self.cap = sc.get("per_trade_risk_cap_pct", 0.25) / 100.0
        self.max_adds = int(sc.get("max_adds_per_stack", 5))
        self.max_trades = int(sc.get("max_trades_per_day", 400))
        self.heat_on = bool(sc.get("heat_guard", {}).get("enabled", True))
        self.atr_mult = float(sc.get("broker_stop", {}).get("atr_mult", 6.0))
        self.point_size = float(sc.get("point_size", 0.01))
        self.contract = float(sc.get("contract_size", 100.0))
        self.probe_units = float(sc.get("probe_lot", 0.01)) * self.contract
        self.trail_keep = float(_goals().get("ratchet", {}).get("trail_keep_frac", 0.25))

        # ----- reward weights: the SAME doctrine -----
        w = dict(_load("rewards"))
        self.w = {k: float(w.get(k, d)) for k, d in {
            "w_net_profit": 6.0, "w_no_drawdown_close": 0.02, "no_drawdown_tolerance": 0.0,
            "w_pyramid_stack_green": 0.20, "w_pullback_with_htf": 0.02,
            "w_idleness_hunger": -0.002, "w_did_nothing": -6.0, "did_nothing_band": 0.25,
            "w_day_goal_hit": 2.0, "day_dd_extra_scale": 0.5, "w_streak_per_day": 0.15,
            "w_trade_consistency": 0.10, "trade_consistency_target": 0.3,
            "w_record_win": 0.25, "w_death_penalty": -10.0}.items()}

        # ----- pull-tag column indices (for the tiny pullback close bonus) -----
        def _idx(names):
            out = [cols.index(n) for n in names if n in cols]
            return torch.tensor(out, dtype=torch.long, device=self.dev)
        self.pull_buy_idx = _idx([f"set{k}::pull_buy" for k in (1, 2, 3, 4)])
        self.pull_sell_idx = _idx([f"set{k}::pull_sell" for k in (1, 2, 3, 4)])

    # ================= episode lifecycle =================
    def reset(self, day_idx, goal, floor, streak_in=None, record_in=None):
        dev = self.dev
        di = torch.as_tensor(day_idx, dtype=torch.long, device=dev)
        N = di.shape[0]
        self.N = N
        z = lambda: torch.zeros(N, self.K, device=dev)
        self.active = z(); self.side = z(); self.units = z(); self.avg = z()
        self.stop = z(); self.bars = z(); self.madv = z(); self.esp = z()
        self.probe = z(); self.adds = z(); self.pull = z()
        self.balance = torch.full((N,), self.eq0, device=dev)
        self.trades_used = torch.zeros(N, device=dev)
        self.dead = torch.zeros(N, dtype=torch.bool, device=dev)
        self.breached = torch.zeros(N, dtype=torch.bool, device=dev)
        self.finalized = torch.zeros(N, dtype=torch.bool, device=dev)
        self.goal = torch.as_tensor(goal, dtype=torch.float32, device=dev).clone()
        self.floor = torch.as_tensor(floor, dtype=torch.float32, device=dev).clone()
        self.ratchet = -self.floor.clone()
        self.day_idx = di
        self.day_len = self.day_lens[di]
        self.streak = (torch.zeros(N, device=dev) if streak_in is None
                       else torch.as_tensor(streak_in, dtype=torch.float32, device=dev).clone())
        self.record = (torch.zeros(N, device=dev) if record_in is None
                       else torch.as_tensor(record_in, dtype=torch.float32, device=dev).clone())
        self.min_eq = torch.zeros(N, device=dev)
        self.max_eq = torch.zeros(N, device=dev)
        self.min_worst = torch.zeros(N, device=dev)   # running min of intrabar worst-case eq% (true breach basis)
        self.cnt_np = torch.zeros(N, device=dev)
        self.wins_np = torch.zeros(N, device=dev)
        self.sum_pnl = torch.zeros(N, device=dev)
        self.sumsq_pnl = torch.zeros(N, device=dev)
        self.best_pnl = torch.zeros(N, device=dev)
        self.day_pnl = torch.zeros(N, device=dev)
        self.goal_hit = torch.zeros(N, dtype=torch.bool, device=dev)
        self.self_hist = torch.zeros(N, self.Lmax, SELF_DIM, device=dev)
        self.t = 0
        ss = self._self_state(0)
        self.self_hist[:, 0] = ss
        return self._build_obs(0)

    # ================= helpers =================
    def _phys(self, t):
        p = self.days_phys[self.day_idx, t]                     # (N,7)
        return (p[:, 0], p[:, 1], p[:, 2], p[:, 3] * self.point_size, p[:, 4],
                p[:, 5], p[:, 6])                               # hi,lo,close,sp,atr,mb,ms

    def _act_mask(self):
        return (self.active > 0.5)

    def _unreal(self, mark):
        a = self._act_mask()
        return (self.side * self.units * (mark[:, None] - self.avg) * a).sum(1)

    def _eq_pct(self, mark):
        return 100.0 * (self.balance + self._unreal(mark) - self.eq0) / self.eq0

    def _open_risk(self, mark, sp):
        a = self._act_mask()
        loss = self.side * (mark[:, None] - self.stop) + sp[:, None]
        tr = torch.clamp(self.units * loss, min=0.0) / self.eq0
        return (tr * a).sum(1)

    def _worst_eq(self, hi, lo, sp):
        a = self._act_mask()
        pxL = torch.where(lo[:, None] <= self.stop, self.stop, lo[:, None]) - sp[:, None]
        pxS = torch.where(hi[:, None] >= self.stop, self.stop, hi[:, None]) + sp[:, None]
        cL = self.units * (pxL - self.avg)
        cS = self.units * (self.avg - pxS)
        contrib = torch.where(self.side > 0, cL, cS) * a
        return 100.0 * (self.balance - self.eq0 + contrib.sum(1)) / self.eq0

    def _flat_cost(self, hi, lo, close, sp):
        a = self._act_mask()
        cL = self.units * ((close[:, None] - lo[:, None]) + sp[:, None])
        cS = self.units * ((hi[:, None] - close[:, None]) + sp[:, None])
        cost = torch.where(self.side > 0, cL, cS) * a
        return 100.0 * cost.sum(1) / self.eq0

    def _self_state(self, t):
        close = self.days_phys[self.day_idx, t, 2]
        sp = self.days_phys[self.day_idx, t, 3] * self.point_size
        a = self._act_mask()
        unreal = self._unreal(close)
        eqp = 100.0 * (self.balance + unreal - self.eq0) / self.eq0
        wr = torch.where(self.cnt_np > 0, self.wins_np / torch.clamp(self.cnt_np, min=1.0),
                         torch.full_like(self.cnt_np, 0.5))
        long_u = (self.units * a * (self.side > 0)).sum(1)
        short_u = (self.units * a * (self.side < 0)).sum(1)
        tot_u = (self.units * a).sum(1)
        open_r = self._open_risk(close, sp)
        maxbars = torch.clamp((self.bars * a).amax(1), max=240.0)
        unreal_pct = unreal / self.eq0 * 100.0
        return torch.stack([
            self.goal / 5.0,
            self.floor / 6.0,
            (self.goal - eqp) / torch.clamp(self.goal, min=1e-6),
            (eqp + self.floor) / torch.clamp(self.floor, min=1e-6),
            torch.clamp(self.ratchet, min=0.0) / 5.0,
            wr,
            torch.clamp(self.streak, max=50.0) / 50.0,
            open_r * 100.0 / 4.0,
            (long_u - short_u) / (1.0 + tot_u),
            maxbars / 240.0,
            self.trades_used / 400.0,
            unreal_pct / 4.0,
        ], dim=1)

    def _build_obs(self, t):
        pos = torch.clamp(torch.arange(t - FRAME + 1, t + 1, device=self.dev), min=0)  # (10,)
        mk = self.days_obs[self.day_idx[:, None], pos[None, :], :]        # (N,10,C)
        sf = self.self_hist[torch.arange(self.N, device=self.dev)[:, None],
                            pos[None, :], :]                              # (N,10,12)
        return torch.cat([mk, sf], dim=2).reshape(self.N, FRAME * (self.C + SELF_DIM))

    # ================= closing =================
    def _realize(self, cmask, frac, px):
        """Close masked slots. cmask (N,K) bool, frac (N,K), px (N,K). -> reward (N,)."""
        cmask = cmask & (self.active > 0.5)
        w = self.w
        units_closed = self.units * frac * cmask
        pnl = self.side * units_closed * (px - self.avg) * cmask
        self.balance = self.balance + pnl.sum(1)
        pnl_pct = 100.0 * pnl / self.eq0
        full = (frac >= 0.999) & cmask
        rr = w["w_net_profit"] * pnl_pct
        nodd = full & (pnl > 0) & (self.probe < 0.5) & (self.madv <= w["no_drawdown_tolerance"])
        rr = rr + w["w_no_drawdown_close"] * nodd.float()
        pyr = full & (self.adds > 0) & (pnl > 0)
        rr = rr + w["w_pyramid_stack_green"] * torch.clamp(self.adds, max=5.0) * pyr.float()
        pll = full & (self.pull > 0.5) & (self.probe < 0.5)
        rr = rr + w["w_pullback_with_htf"] * pll.float()
        r = (rr * cmask).sum(1)
        # closed-trade stats (non-probe) for win-rate / consistency / record
        npb = cmask & (self.probe < 0.5)
        self.cnt_np = self.cnt_np + npb.sum(1)
        self.wins_np = self.wins_np + (npb & (pnl > 0)).sum(1)
        self.sum_pnl = self.sum_pnl + (pnl_pct * npb).sum(1)
        self.sumsq_pnl = self.sumsq_pnl + ((pnl_pct ** 2) * npb).sum(1)
        self.best_pnl = torch.maximum(self.best_pnl,
                                      torch.where(cmask, pnl_pct, torch.full_like(pnl_pct, NEG)).amax(1))
        # position update: half -> scale units; full -> clear slot
        keep = 1.0 - frac
        self.units = torch.where(cmask & ~full, self.units * keep, self.units)
        for fld in ("active", "side", "units", "avg", "stop", "bars", "madv", "esp", "probe", "adds", "pull"):
            t_ = getattr(self, fld)
            setattr(self, fld, torch.where(full, torch.zeros_like(t_), t_))
        return r

    # ================= one bar =================
    def step(self, op, size):
        dev = self.dev
        op = torch.as_tensor(op, dtype=torch.long, device=dev)
        size = torch.as_tensor(size, dtype=torch.float32, device=dev)
        live = ~self.dead & ~self.finalized
        op = torch.where(live, op, torch.zeros_like(op))
        risk = torch.clamp(size, 0.05, 1.0) * self.cap

        # ---- pull tags at the DECISION bar (before advancing) ----
        t0 = self.t
        row0 = self.days_obs[self.day_idx, t0]        # (N,C) ONE gather (review: was materializing the whole day every bar)
        pull_buy = (row0[:, self.pull_buy_idx] > 0).any(1) \
            if self.pull_buy_idx.numel() else torch.zeros(self.N, dtype=torch.bool, device=dev)
        pull_sell = (row0[:, self.pull_sell_idx] > 0).any(1) \
            if self.pull_sell_idx.numel() else torch.zeros(self.N, dtype=torch.bool, device=dev)

        # ---- advance one bar ----
        self.t += 1
        t = self.t
        hi, lo, close, sp, atr, mb, ms = self._phys(t)
        reached_end = (t >= self.day_len - 1)
        act_mask = live & ~reached_end
        r = torch.zeros(self.N, device=dev)

        # ---- shell pieces shared by open/add ----
        atr_bad = ~torch.isfinite(atr) | (atr <= 0)
        budget_block = self.trades_used >= self.max_trades
        mark = close
        eff = torch.maximum(-self.floor, self.ratchet)
        dist = torch.clamp((self._eq_pct(mark) - eff) / 100.0, min=0.0)
        open_r = self._open_risk(mark, sp)

        # ===== OPEN (1,2,9,10) =====
        open_long = (op == 1) | (op == 9)
        open_short = (op == 2) | (op == 10)
        probe = (op == 9) | (op == 10)
        oside = torch.where(open_long, 1.0, torch.where(open_short, -1.0, 0.0))
        mask_block = ((oside > 0) & (mb > 0)) | ((oside < 0) & (ms > 0))
        heat_block = self.heat_on & (open_r + risk > dist + 1e-12)
        free = self.active < 0.5
        has_free = free.any(1)
        free_idx = torch.argmax(free.float(), 1)
        ok_open = (act_mask & (open_long | open_short) & ~mask_block & ~budget_block
                   & ~atr_bad & ~heat_block & has_free)
        stop_dist = self.atr_mult * atr
        fill = torch.where(oside > 0, hi + sp, lo - sp)
        units = risk * self.eq0 / (stop_dist + sp)
        units = torch.where(probe, torch.minimum(units, torch.full_like(units, self.probe_units)), units)
        pull_tag = torch.where(open_long, pull_buy, torch.where(open_short, pull_sell,
                                                                torch.zeros_like(pull_buy)))
        e = ok_open.nonzero(as_tuple=True)[0]
        if e.numel():
            si = free_idx[e]
            self.active[e, si] = 1.0
            self.side[e, si] = oside[e]
            self.units[e, si] = units[e]
            self.avg[e, si] = fill[e]
            self.stop[e, si] = fill[e] - oside[e] * stop_dist[e]
            self.esp[e, si] = sp[e]
            self.probe[e, si] = probe[e].float()
            self.bars[e, si] = 0.0
            self.madv[e, si] = 0.0
            self.adds[e, si] = 0.0
            self.pull[e, si] = pull_tag[e].float()
            self.trades_used[e] += 1.0

        # ===== ADD (3,4) — biggest same-side non-probe winner =====
        add_long = (op == 3); add_short = (op == 4)
        aside = torch.where(add_long, 1.0, torch.where(add_short, -1.0, 0.0))
        same = (self.active > 0.5) & (self.side == aside[:, None]) & (self.probe < 0.5)
        winner = same & (aside[:, None] * (mark[:, None] - self.avg) > 0) & (self.adds < self.max_adds)
        big = torch.where(winner, self.units, torch.full_like(self.units, NEG)).argmax(1)
        has_t = winner.any(1)
        a_mask_block = ((aside > 0) & (mb > 0)) | ((aside < 0) & (ms > 0))
        a_heat = self.heat_on & (open_r + risk > dist + 1e-12)
        ok_add = (act_mask & (add_long | add_short) & has_t & ~a_mask_block
                  & ~budget_block & ~atr_bad & ~a_heat)
        e = ok_add.nonzero(as_tuple=True)[0]
        if e.numel():
            si = big[e]
            fillA = torch.where(aside[e] > 0, hi[e] + sp[e], lo[e] - sp[e])
            dstop = aside[e] * (fillA - self.stop[e, si]) + sp[e]
            good = dstop > 0
            e, si, fillA, dstop = e[good], si[good], fillA[good], dstop[good]
            if e.numel():
                add_u = risk[e] * self.eq0 / dstop
                new_u = self.units[e, si] + add_u
                self.avg[e, si] = (self.units[e, si] * self.avg[e, si] + add_u * fillA) / new_u
                self.units[e, si] = new_u
                self.adds[e, si] += 1.0
                self.trades_used[e] += 1.0

        # ===== CLOSE (5,6,7,8) — biggest same-side incl. probes =====
        cl_hl = (op == 5); cl_l = (op == 6); cl_hs = (op == 7); cl_s = (op == 8)
        cside = torch.where(cl_hl | cl_l, 1.0, torch.where(cl_hs | cl_s, -1.0, 0.0))
        frac_env = torch.where(cl_hl | cl_hs, 0.5, 1.0)
        csame = (self.active > 0.5) & (self.side == cside[:, None])
        cbig = torch.where(csame, self.units, torch.full_like(self.units, NEG)).argmax(1)
        ok_close = act_mask & (cl_hl | cl_l | cl_hs | cl_s) & csame.any(1)
        cmask = torch.zeros(self.N, self.K, dtype=torch.bool, device=dev)
        fracK = torch.ones(self.N, self.K, device=dev)
        e = ok_close.nonzero(as_tuple=True)[0]
        if e.numel():
            cmask[e, cbig[e]] = True
            fracK[e, cbig[e]] = frac_env[e]
        pxK = torch.where(self.side > 0, (lo - sp)[:, None], (hi + sp)[:, None])
        r = r + self._realize(cmask, fracK, pxK)

        acted = ok_open | ok_add | ok_close

        # ===== broker stops (gap-aware) =====
        stopL = (self.active > 0.5) & (self.side > 0) & (lo[:, None] <= self.stop)
        stopS = (self.active > 0.5) & (self.side < 0) & (hi[:, None] >= self.stop)
        pxL = torch.where(hi[:, None] >= self.stop, self.stop, lo[:, None]) - sp[:, None]
        pxS = torch.where(lo[:, None] <= self.stop, self.stop, hi[:, None]) + sp[:, None]
        pxBK = torch.where(self.side > 0, pxL, pxS)
        r = r + self._realize(stopL | stopS, torch.ones(self.N, self.K, device=dev), pxBK)

        # ===== marks: bars, adverse, equity min/max =====
        a = self._act_mask()
        self.bars = self.bars + a.float()
        new_madv = torch.maximum(self.madv, self.side * (self.avg - close[:, None]) - self.esp)
        self.madv = torch.where(a, new_madv, self.madv)
        eqp = self._eq_pct(close)
        self.min_eq = torch.minimum(self.min_eq, eqp)
        self.max_eq = torch.maximum(self.max_eq, eqp)

        # ===== floor LAW (intrabar worst-case) =====
        worst = self._worst_eq(hi, lo, sp)
        self.min_worst = torch.minimum(self.min_worst, worst)   # true distance-to-breach for the surrogate
        stand = live & ((worst <= eff) | (eqp <= eff))
        if stand.any():
            fpx = torch.where(self.side > 0, (lo - sp)[:, None], (hi + sp)[:, None])
            fcmask = a & stand[:, None]
            r = r + self._realize(fcmask, torch.ones(self.N, self.K, device=dev), fpx)
            self.breached = self.breached | (stand & (torch.minimum(worst, eqp) <= -self.floor))
            self.dead = self.dead | stand

        # ===== ratchet =====
        # 2026-07-20 LAW FIX (lift demo finding): the lock line must INCLUDE the flatten
        # cost, or the stand-down flatten realizes goal - cost (banked ~2.9% on a 3.0%
        # target — the day "reaches the target" but never banks it). R3#7's documented
        # intent is "flatten still realizes >= goal"; arming already requires
        # eq - flat_cost >= goal, so goal + flat_cost is at-or-below equity when set.
        flat_cost = self._flat_cost(hi, lo, close, sp)
        cond = (eqp - flat_cost >= self.goal) | (self.ratchet >= self.goal)
        trail = self.max_eq - self.floor * self.trail_keep
        newr = torch.maximum(torch.maximum(self.ratchet, self.goal + flat_cost), trail)
        self.ratchet = torch.where(live & cond & ~self.dead, newr, self.ratchet)

        # ===== idleness hunger (approx: flat & hold) =====
        flat_hold = live & (self._act_mask().sum(1) < 0.5) & (op == 0) & ~acted
        r = r + self.w["w_idleness_hunger"] * flat_hold.float()

        # ===== done / day-end =====
        done = self.dead | reached_end
        just = done & ~self.finalized & live
        if just.any():
            # midnight flatten for non-dead reached_end envs
            need_flat = just & ~self.breached & (self._act_mask().any(1))
            if need_flat.any():
                fpx = torch.where(self.side > 0, (lo - sp)[:, None], (hi + sp)[:, None])
                r = r + self._realize(self._act_mask() & need_flat[:, None],
                                      torch.ones(self.N, self.K, device=dev), fpx)
            r = r + self._day_end_reward(just)
            self.finalized = self.finalized | just

        # store self-state for the NEXT obs and build it
        nt = min(t, self.Lmax - 1)
        if t < self.Lmax:
            self.self_hist[:, t] = self._self_state(t)
        obs = self._build_obs(nt)
        done_all = self.dead | self.finalized | (t >= self.Lmax - 1)
        return obs, r, done_all

    def _day_end_reward(self, just):
        w = self.w
        day_pnl = 100.0 * (self.balance - self.eq0) / self.eq0
        gh = (day_pnl >= self.goal) & ~self.breached
        self.day_pnl = torch.where(just, day_pnl, self.day_pnl)
        self.goal_hit = torch.where(just, gh, self.goal_hit)
        rr = torch.zeros(self.N, device=self.dev)
        did_nothing = (self.cnt_np == 0) | (day_pnl.abs() < w["did_nothing_band"])
        rr = rr + w["w_did_nothing"] * did_nothing.float()
        rr = rr + w["w_death_penalty"] * self.breached.float()
        ghf = gh.float()
        rr = rr + w["w_day_goal_hit"] * ghf
        dd_extra = (w["w_day_goal_hit"]
                    * torch.clamp(1.0 + self.min_eq / torch.clamp(self.floor, min=1e-6), min=0.0)
                    * w["day_dd_extra_scale"])
        rr = rr + dd_extra * ghf
        self.streak = torch.where(gh, self.streak + 1.0, torch.zeros_like(self.streak))
        rr = rr + w["w_streak_per_day"] * self.streak * ghf
        mean = self.sum_pnl / torch.clamp(self.cnt_np, min=1.0)
        var = torch.clamp(self.sumsq_pnl / torch.clamp(self.cnt_np, min=1.0) - mean ** 2, min=0.0)
        spread = torch.sqrt(var)
        cons = (self.cnt_np > 1).float()
        rr = rr + w["w_trade_consistency"] * torch.clamp(w["trade_consistency_target"] - spread, min=0.0) * cons
        newrec = gh & (self.best_pnl > self.record)
        self.record = torch.where(newrec, self.best_pnl, self.record)
        rr = rr + w["w_record_win"] * newrec.float()
        return rr * just.float()

    def results(self):
        # clone so callers can hold the dict across a later rollout (review: no aliasing)
        return {"day_pnl": self.day_pnl.clone(), "goal_hit": self.goal_hit.clone(),
                "breached": self.breached.clone(), "streak": self.streak.clone(),
                "record": self.record.clone(), "min_eq": self.min_eq.clone(),
                "min_worst": self.min_worst.clone(),
                "target": self.goal.clone(), "risk": self.floor.clone()}


# --------------------- tiny smoke test (random policy) ---------------------
if __name__ == "__main__":
    import numpy as np
    from core.configs import path as rpath
    from training.gpu_data import build_day_tensors
    src = sys.argv[1] if len(sys.argv) > 1 else rpath("data", "XAUUSD_M1_drill.csv")
    tag = os.path.splitext(os.path.basename(src))[0]
    do, dp, dl, dates, cols = build_day_tensors(src, cache_path=rpath("artifacts", "gpu_cache_%s.npz" % tag))
    sim = FastSim(do, dp, dl, cols, device="cpu", K=48)
    torch.manual_seed(0)
    N = 16
    day_idx = torch.randint(0, do.shape[0], (N,))
    goal = torch.empty(N).uniform_(2.5, 70.3)
    floor = torch.empty(N).uniform_(1.0, 4.4)
    obs = sim.reset(day_idx, goal, floor)
    print("obs shape:", tuple(obs.shape), "| expected (N,1820)")
    steps = 0
    while True:
        op = torch.randint(0, 11, (N,))
        size = torch.rand(N)
        obs, r, done = sim.step(op, size)
        steps += 1
        if bool(done.all()) or steps > sim.Lmax:
            break
    res = sim.results()
    print("steps:", steps)
    print("day_pnl  :", np.round(res["day_pnl"].numpy(), 2))
    print("breached :", res["breached"].numpy().astype(int))
    print("goal_hit :", res["goal_hit"].numpy().astype(int))
    print("reward finite:", bool(torch.isfinite(r).all()))
