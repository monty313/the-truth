# #1 — TEACH META + RL TO HIT THE GOAL **EVERY DAY**

**Priority:** NUMBER ONE. Everything else is secondary.  
**Style:** Brutally honest. Checklist only. No motivation theater.  
**Goal (one line):** On **every** scored day, under **one frozen policy**, clear the typed target% without breach — the way **Mark** would on the same multi-TF chart — with random T%/R%, **no retrain** at test time.

| Field | Value |
|-------|--------|
| **Repo** | `the-truth` · lineage `lineages/adaptive_rl_brain_7_31_26/` |
| **Doctrine** | `POLICY_EQUALS_MARK_ON_CHART.md` · pt5 · `SOUL_MATCH.md` |
| **Recreate bible** | `PROGRESS_RECREATE__POLICY_EQUALS_MARK__2026-08-05.md` |
| **Meta law** | Probe **practice** · adopt only on **forward** consistency |
| **PROVEN** | Never overwrite |

---

## 0) Brutal truth first (read before any train)

- [ ] **Accept this:** The net does not “understand Mark.” It copies statistics. If the statistics are from the wrong path, it will fail every day.
- [ ] **Accept this:** Meta does not invent chart truth. Meta only searches **allowed dials/hparams** and must be judged on **forward** days.
- [ ] **Accept this:** “Loss went down” is not the goal. **Day clear like Mark, breach 0, pack not worse** is the goal.
- [ ] **Accept this:** Rewards are a **megaphone**, not a mind. Cranking them without path-matched Mark labels creates thrash or freeze.
- [ ] **Accept this:** One policy for all days. Retrain-per-day is cheating and is **not** long consistency.
- [ ] **Accept this:** If practice improves and forward dies, you taught **costume**, not skill. Discard it.
- [ ] **Accept this:** 92 signal agents are **clues**. Mark / force / regime owns **side**. Never promote agents to soul.
- [ ] **Accept this:** Current honest state (2026-08-05): Mark soul **50/50** on frozen pack; policy best **~33/50** same_outcome; gap is path-matched transfer, not “Mark won’t answer.”

**If any box above is rejected, stop training. You will thrash compute.**

---

## 1) Define “goal every day” (machine language)

### 1.1 Single-day success (the atom)

- [ ] Typed **target%** and **risk%** are runtime inputs only (not baked into weights as the only pair).
- [ ] **Clear** = end equity% ≥ target% **AND** never touched −risk% (breach) that day.
- [ ] **Mark-clear** = full-day soul plan would clear (teacher yardstick).
- [ ] **Policy-clear** = embryo under pure_greedy + mark_align clears.
- [ ] **same_outcome** = policy-clear bit equals Mark-clear bit.
- [ ] **Thrash fail** = many more entries than Mark with no better clear (noise, not skill).
- [ ] **HOLD skill** = wait when Mark waits (slingshot load), not “I don’t know.”

### 1.2 Every-day success (the product)

- [ ] On the **frozen recipe** window (start with 50 days), target:
  - [ ] `same_outcome == N/N` (or policy_clear == mark_clear == N)
  - [ ] `n_breach == 0`
  - [ ] Same **one** checkpoint for all days (no retrain between days)
- [ ] On **forward** holdout (days after practice): same law — clear rate / streak up only if real.
- [ ] On **random T/R** from `ten_pairs.json` (seed fixed for compare, soft_bias false for honesty).

### 1.3 Frozen score recipe (do not move the goalposts)

- [ ] Data: `XAUUSD_curriculum_2026.csv`, min_bars ≥ 900  
- [ ] Window: first 50 loadable days (`2026-01-20` → `2026-03-30`) for the 50d pack  
- [ ] Seed: **42** · `soft_bias=false`  
- [ ] Decode: **full_obs** + **mark_align** + **pure_greedy**  
- [ ] Eyes: `mark_doctrine` · `mark_soul=True`  
- [ ] Ckpt under test: embryo only (e.g. `mark_clone_full_obs_v1.pt`)  
- [ ] PROVEN not loaded into Mark-obs  

---

## 2) Split the brain: what META learns vs what RL/POLICY learns

### 2.1 What the **RL / policy model** must learn (the student)

Teach it to **act** like Mark on the **live path**:

| # | Skill | How you teach it | How you know it failed |
|---|--------|------------------|------------------------|
| R1 | HTF permission (side) | Labels + force-gate; never label against tide | Opens against force; wrong_side_rate hot |
| R2 | LTF timing only | Entries only on slingshot resume labels | Fires mid-breath; thrash entries |
| R3 | Intentional HOLD | HOLD labels when Mark waits | Hold-rate → 0 then breach, or freeze forever |
| R4 | Size / adds (soul) | Soul-plan labels (goal-relative) | Clears chart side but misses target size |
| R5 | Kill tide | Flatten / no add when force breaks | Averages into floor |
| R6 | Capital danger | danger ≥ ~0.45 no new risk | Late revenge opens |
| R7 | Pack consistency | Keep only if full window same↑ | One day wins, 10 days die |

### 2.2 What **meta-learning** must learn (the coach)

Teach meta to **search training conditions**, not to be Mark:

| # | Skill | How you teach it | How you know it failed |
|---|--------|------------------|------------------------|
| M1 | Where to search | Wrong-side → TREND_KNOBS; weak forward → CONSISTENCY_FORWARD_KNOBS | Random dial thrash every gen |
| M2 | How hard to search | Aggressive scale only when disease hot | Always max noise → no stable champion |
| M3 | Adopt honesty | **Forward** consistency must improve | Practice-only KEPT champions |
| M4 | Breach conscience | Never adopt if forward breach rises | “Higher clear” with more deaths |
| M5 | Streak conscience | Forward longest day-streak not shorter | Clear% flat luck, streak dies |
| M6 | Practice screen | Reject if practice collapses (forward fluke) | Overfit forward slice |
| M7 | Allowed surface only | Reward/hparams in BOUNDS; streak dials pack | Shell/PROVEN/entry-law edits |

### 2.3 What **neither** is allowed to “learn”

- [ ] Shell physics rewrite (heat/bank/breach code) via search  
- [ ] Trail + cushion + scale-in package  
- [ ] PROVEN overwrite  
- [ ] Second personality / thrash teacher as side owner  
- [ ] Fitting weights or dials on **forward** labels  
- [ ] Calling BC match rate “done” without day same_outcome  

---

## 3) The only teaching loop that works (both systems)

```
Mark chart truth (soul plan / HITL)
        ↓ path-matched bar labels (same gates as live)
RL/BC/DAgger on PRACTICE only
        ↓
Score PRACTICE (screen) + FORWARD (judge) + FROZEN pack (50d same)
        ↓
KEEP weights only if pack/forward not worse
        ↓
Meta proposes dial/hparam delta (allowed set only)
        ↓
Short probe on PRACTICE
        ↓
ADOPT config only if FORWARD consistency improves
        ↓
Repeat until same_outcome → N/N and forward holds
```

- [ ] Print this loop on the wall. Do not invent a parallel loop.

---

## 4) Checklist: teach the **RL model** every day (primary)

### 4.0 Preconditions (every session)

- [ ] `PYTHONPATH=.;code` from the-truth root  
- [ ] Embryo path known; PROVEN not in load list for Mark-obs  
- [ ] Force-gate + thrash caps + full_obs code present  
- [ ] Gate bug fixed: no `"flat" in regime` false positive on `flat_undefined`  
- [ ] Oracle / soul plans available or regenerable for the window  
- [ ] Baseline frozen once: write `BASELINE_*__frozen.json` and **never rewrite history**  

### 4.1 Build Mark’s answer key (teacher) — before any gradient

- [ ] For each day in the train window: run **Mark soul plan** (full chart, size, force adds).  
- [ ] Record: clear?, breach?, n_entries, side path, pnl.  
- [ ] Class misses: `MARK_WOULD_TAKE` vs `NO_OPPORTUNITY` vs `AWARD`.  
- [ ] **Do not** train thrash on NO_OPPORTUNITY days.  
- [ ] Cache oracle to disk (`MARK_ORACLE_CACHE__50d.json`) so you do not re-search Mark every gen.  
- [ ] Verify teacher: on frozen 50d pack Mark should be **~50/50** clear (if not, teacher code is broken — fix doctrine/soul before policy).

### 4.2 Path-match the labels (the non-negotiable)

- [ ] Live policy decode path = label path:  
  - [ ] same `full_obs`  
  - [ ] same `mark_align` / force-gate  
  - [ ] same thrash caps  
  - [ ] same pure_greedy at score time  
- [ ] If soul plan says SELL-add but live path would HOLD:  
  - [ ] **Either** fix live path to allow Mark’s legal action  
  - [ ] **Or** label HOLD under live path (honest)  
  - [ ] **Never** average them and hope.  
- [ ] DAgger: collect labels on **states the policy actually visits**, not only on Mark’s perfect walk.

### 4.3 Build the day’s curriculum (what to show the net)

For **each** MARK_WOULD_TAKE day (priority queue: worst policy PnL first):

- [ ] Extract bar rows: obs (168-dim) + Mark action under path-matched rules.  
- [ ] **Oversample directional** labels (BUY/SELL) ~4–6× vs raw HOLD (lab lesson: HOLD-heavy → dir_match ~0.09).  
- [ ] Keep enough HOLD that pred_hold_rate does not collapse (~0.3–0.5 ballpark; if → 0.05, expect breach).  
- [ ] Optional: light **award-day self-imitate** so winning days do not forget.  
- [ ] Sample weights from streak dials (MWT miss penalty, soul-side entry bonus, thrash break) — **after** labels are correct, not instead of them.

### 4.4 Train one surgical step (not a religion)

- [ ] Warm-start from **current best embryo** only.  
- [ ] BC epochs moderate (e.g. 20–40); watch loss but **do not trust it**.  
- [ ] **KL anchor** to best embryo (coef ~0.5–0.6 when hold-rate fragile).  
- [ ] One miss day heavy focus **or** small batch of MWT days — not all history every time if it thrashs.  
- [ ] Never mix a second teacher (thrash agent spam) into side labels.

### 4.5 Score like a prosecutor (same day discipline)

- [ ] Re-score **entire frozen pack** (50d recipe), not just the focus day.  
- [ ] Record: same_outcome, policy_clear, mark_clear, n_breach, mwt count, max streak if available.  
- [ ] Dual score once if claiming a KEEP (run score twice; agree).  
- [ ] Optional forward score if this train touched dials/meta.

### 4.6 KEEP / REJECT (conscience — no vibes)

**KEEP only if ALL true:**

- [ ] `n_breach == 0` (or ≤ champion breach and still 0 preferred)  
- [ ] `same_outcome >= best.same_outcome`  
- [ ] `policy_clear >= baseline.policy_clear` (floor, e.g. 27)  
- [ ] Focus day converted **or** pack same rose  
- [ ] pred_hold not in death zone if breach risk appeared historically  

**Else:**

- [ ] **REJECT**  
- [ ] Restore previous best weights bit-exact  
- [ ] Log why (breach / same down / hold collapse / path mismatch)  
- [ ] Do **not** keep “because dir_match was 0.96”

### 4.7 Daily RL operator checklist (literal calendar day)

- [ ] Load best embryo + baseline meters.  
- [ ] List remaining MWT dates.  
- [ ] Train **one** focus day (or one small batch).  
- [ ] Full pack score → KEEP/REJECT.  
- [ ] Update `LEARNING_*.md` / BEST json.  
- [ ] If 3 REJECT in a row on same day: stop cranking rewards; run path autopsy (plan vs live action).  
- [ ] If HOLD collapse: HOLD-repair BC + raise KL; do not push entry dials.  
- [ ] If entry never fires on MWT: check force-gate false block; then directional oversample.  
- [ ] Stop when same_outcome hits pack size or human calls stop.

### 4.8 Teach “every day” behavior inside the reward (only after 4.2)

Use **streak pack** dials (searchable; not shell):

- [ ] `streak_award_base` / `streak_award_per_prior` — pay chains of clears  
- [ ] `streak_break_penalty` — punish breaking a live streak  
- [ ] `mark_would_take_eod_penalty` — punish missing a Mark-clearable day  
- [ ] `no_opp_hold_bonus` — pay honest wait when no force path  
- [ ] `soul_side_entry_bonus` / `soul_side_misread_penalty` — side alignment on MWT days  

- [ ] Update dials only from **autopsy**, not vibes.  
- [ ] After dial change: must re-train with labels or sample weights, then **full pack + forward** gate.  
- [ ] Roll back dials with weights if gate fails.

---

## 5) Checklist: teach **meta-learning** every day (coach)

### 5.0 Meta constitution (non-negotiable)

- [ ] Meta **never** trains on forward days.  
- [ ] Meta **never** adopts because practice clear% looked good alone.  
- [ ] Meta **never** edits shell or PROVEN.  
- [ ] Meta **never** replaces Mark labels.  
- [ ] Champion score **is** forward consistency (clear fraction on holdout).  
- [ ] Secondary: forward breach not worse; forward longest streak not shorter; side veto; practice screen.

### 5.1 Wire the pools (every meta run)

- [ ] `practice_days` = chronological first N (e.g. 50)  
- [ ] `forward_days` = next M (e.g. 40) — empty forward ⇒ **refuse to run** if `require_forward=True`  
- [ ] Common random numbers: same generator seed for champ vs candidate score  
- [ ] Day-cluster honesty: do not let one lucky day fake significance (paired gates)

### 5.2 What meta is allowed to mutate

**Production `meta_tuner` BOUNDS (examples):**

- [ ] Mark trend: `w_pullback_with_htf`, `w_with_trend_close`, `w_against_trend_close`, `w_quick_pull_close`, `w_setup_skip`  
- [ ] Consistency: `w_day_goal_hit`, `w_streak_per_day`, `w_trade_consistency`, `w_did_nothing`, `w_death_penalty`, `w_net_profit`  
- [ ] Learnability: `lr`, `entropy_coef`  

**Lineage streak dials (Mark pack):**

- [ ] Only keys in `STREAK_REWARD_DIALS` in `rewards.py`  
- [ ] Clip with `clip_streak_dials`  

### 5.3 Adaptive search (disease → where to look)

- [ ] If wrong_side / side_bias hot → force **TREND_KNOBS**, aggressive scale  
- [ ] If forward consistency weak or streak weak → force **CONSISTENCY_FORWARD_KNOBS**  
- [ ] If both → merge force groups  
- [ ] If cool → small normal mutations  
- [ ] Sticky focus for a few gens after disease clears (do not instantly go timid)

### 5.4 One meta generation (checklist)

- [ ] Snapshot champion config + brain weights  
- [ ] `mutate` ≤ K knobs (few-at-a-time)  
- [ ] **Probe** short PPO/BC only on **practice**  
- [ ] Score **forward** (consistency, breach, longest_streak, wrong_side)  
- [ ] Score **practice sample** (screen only)  
- [ ] `forward_adopt_ok`?  
- [ ] `practice_screen_ok`?  
- [ ] If yes: champion = candidate; log adopt  
- [ ] If no: **restore brain + config**; log reject reasons  
- [ ] Never leave a rejected net loaded as “current”

### 5.5 Daily meta operator checklist

- [ ] Run meta only when RL path is stable enough to score (else meta optimizes noise).  
- [ ] Cap wall time; keep history JSON of adopts/rejects.  
- [ ] If 10 rejects in a row: disease is **not dial-shaped** — go back to labels/path (section 4).  
- [ ] Export champion dials to `STREAK_REWARD_DIALS__latest.json` / rewards.yaml only after forward KEEP.  
- [ ] Human LIVE gate still required before real money (lab auto-adopt ≠ live).

### 5.6 How meta helps “every day” without lying

Meta’s job toward everyday clears:

- [ ] Increase pressure to **hit day goal** (`w_day_goal_hit`) without raising breach  
- [ ] Pay **streak continuity** (`w_streak_per_day` / streak pack) so the student hates breaking chains  
- [ ] Pay **trade consistency** (not random flick)  
- [ ] Punish death / floor  
- [ ] Cure wrong-side with trend knobs when side rulers hot  
- [ ] **Stop** when forward meters plateau — hand back to RL label surgery  

---

## 6) Combined daily schedule (run this as the #1 ritual)

### Morning — measure truth

- [ ] Score current best embryo on frozen 50d recipe → print same / clear / breach / mwt  
- [ ] Score forward holdout if meta changed anything yesterday  
- [ ] Diff vs BEST; if drifted without KEEP log → restore  

### Midday — teach RL (student)

- [ ] Pick top 1–3 MWT days  
- [ ] Path-matched labels + dir oversample + KL  
- [ ] Train → full pack KEEP/REJECT  
- [ ] Write learning line to `LEARNING_*.md`  

### Afternoon — teach meta (coach) only if student stable

- [ ] If breach 0 and same not collapsing: meta gens on practice/forward  
- [ ] Adopt dials only on forward win  
- [ ] Feed new dials into next RL sample weights  

### Evening — honesty report

- [ ] One table: same, clear, mwt, breach, max_streak, forward_consistency  
- [ ] One sentence: what path failed (labels / gate / thrash / dial)  
- [ ] One next experiment only (not five)  
- [ ] PROVEN mtime unchanged? yes/no  

---

## 7) Failure playbooks (if “every day” is not happening)

### 7.1 Many days miss, Mark would take (MWT high)

- [ ] Confirm Mark plans still 50/50 (teacher healthy)  
- [ ] Path autopsy: live action vs plan on miss bars  
- [ ] One-day DAgger + KL; do not multi-day thrash BC  
- [ ] Dir oversample; check force-gate not false-blocking  

### 7.2 Breach appears

- [ ] REJECT immediately; restore best  
- [ ] HOLD-repair; lower entry pressure; raise KL  
- [ ] Check thrash caps still on  
- [ ] Do not “fix with more reward death” first — stop firing  

### 7.3 High dir_match, same_outcome flat

- [ ] You optimized the **wrong exam** (bars ≠ days)  
- [ ] Switch objective pressure to day clear / same_outcome keep-reject  
- [ ] Align label path to live decode  
- [ ] Stop celebrating match_rate  

### 7.4 Practice up, forward down

- [ ] Discard candidate (meta + weights)  
- [ ] Shrink train window leakage  
- [ ] More forward weight on adopt gate (already mandatory)  
- [ ] Audit: any forward day in train set? must be ∅  

### 7.5 Meta adopts thrash configs

- [ ] Side veto + breach gate not enforced — fix gates  
- [ ] Force TREND_KNOBS when wrong_side hot  
- [ ] Reduce mutation scale  

### 7.6 “Rewards will save us”

- [ ] Read `MARK_HERE_ANSWER__why_rewards_alone.md`  
- [ ] Return to section 4.2 path-matched labels  
- [ ] Autopsy dials only after labels exist  

---

## 8) Done criteria (when you may stop calling it #1 fire drill)

- [ ] Frozen pack: `same_outcome == 50/50` (or policy_clear == mark_clear == 50), breach 0  
- [ ] Dual re-score agrees  
- [ ] Forward holdout: consistency not worse than champion claim; breach 0 preferred  
- [ ] Random T/R still works (seed change smoke test) without retrain  
- [ ] PROVEN untouched  
- [ ] Learning log + BEST json + this checklist all updated  
- [ ] Human Mark can open MARK HERE and not contradict the day diary on spot checks  

**Until then:** this checklist is the #1 work. Not new indicators. Not more agents. Not PROVEN edits.

---

## 9) Commands (cheat sheet)

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"

# Measure pack (student truth)
python lineages/adaptive_rl_brain_7_31_26/fable_50d_one_day.py
# or rapid / mark_match_loop / test_run_10d_mark_vs_policy.py --full-obs

# Autopsy → dials
python lineages/adaptive_rl_brain_7_31_26/autopsy_streak_gaps.py

# Meta under forward law (coach)
python lineages/adaptive_rl_brain_7_31_26/meta_forward_consistency.py --dry-score
python tests/test_self_heal_mri.py

# Army Mark+KAG mentor (soul dialogue)
cd C:\Users\user\OneDrive\Desktop\ARMY\01_SYSTEM
$env:PYTHONPATH = "packages\core"
python scripts\cycles\fable5_mark_here_kag_cycle.py
```

---

## 10) One page for the wall

1. **Mark answers the chart.**  
2. **Labels must use the bot’s live rules.**  
3. **Train on practice.**  
4. **Judge on forward + full frozen pack.**  
5. **KEEP only if same↑ and breach 0.**  
6. **Meta only moves allowed dials; same judge.**  
7. **Reject restores the last honest champion.**  
8. **Repeat until every day is Mark’s day under one policy.**

---

*If this file conflicts with vibes, this file wins. If it conflicts with `POLICY_EQUALS_MARK_ON_CHART.md` shell/PROVEN laws, the policy law wins and this file must be patched.*
