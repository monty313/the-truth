# THE BOT'S ISSUE AND ITS CURE — the court-ordered homework
### Argued under IRAC · sealed 2026-07-21 · doctrine/flea-jar (precedence)
*The court's order: identify what was THE BOT'S issue — not the trainer's — provide the
cure with evidence, and make recurrence impossible. "I don't want to see you in this
courtroom again for the same issue." Understood, Your Honor. Here is the case that ends
the case.*

---

## I — ISSUE

The patient, Momentum One, was **physically capable** (Part One: the money is in the
market; the +6.53% day is frozen proof), **perfectly safe** (zero floor breaches across
every training run ever logged), and **active** (~16 trades/day measured). Yet it was
**flat and timid**: 1 day cleared out of 90, +0.04% per day, and total stillness on the
richest days. The question before the court: **what was actually wrong with the bot?**

## R — RULE

One law of reinforcement learning governs, and this court now adopts it as doctrine:

> **THE MIRROR LAW.** A learning bot becomes *exactly* what its feedback measures — never
> what its owners intend. Therefore every apparent "bot deficiency" is a **measurement
> deficiency** until proven otherwise. The charge sheet for a broken bot must begin with
> its scoreboard.

Corollary (the recurrence bar): a measurement deficiency is cured only when it is
(1) fixed, (2) **pinned by a test that fails loudly if it returns**, and (3) guarded by a
standing process that hunts its siblings. Fixes without pins are promises; this court
accepts only machinery.

## A — APPLICATION

The bot's medical chart shows **four infections — every one a feedback disease, none a
disease of the bot's body.** The bot obeyed its scoreboard faithfully; the scoreboard lied.

**Infection 1 — the pay-cliff at the finish line.** The win-lock (ratchet) parked every
good day at goal-minus-fee (+2.9% on a 3.0% target), and the strict day-end judge paid
**zero** for +2.97%. The bot reached the target thousands of times and was paid *nothing*
— so it learned, correctly, that reaching the target is worthless. Its stillness was not
cowardice; it was **obedience**.
*Evidence:* the law fix in both engines ([training/fastsim.py](https://github.com/monty313/the-truth/blob/main/training/fastsim.py),
[backtesting/simulator.py](https://github.com/monty313/the-truth/blob/main/backtesting/simulator.py), search "LAW FIX", commit `6ed20a7`);
measured effect of the fix alone: 1/90 → 3/90 cleared; with retraining: **+3.07% and
+4.59% banked** the same day ([HANDOFF](https://github.com/monty313/the-truth/blob/main/HANDOFF_2026-07-20.md) §2–3).

**Infection 2 — counterfeit applause.** A batch bug paid streak credit to **every**
simulated market whenever **any one** of them finished its day — noise rewarded as
achievement, records inflated 10–100×, including the counterfeit "days in a row" numbers
of the old era. The bot was cheered for nothing; nothing is what it learned to do.
*Evidence:* fix + dissection in commit `86a2ae4`; the streak law now **pinned by a
regression test** ([tests/test_ratchet_and_streak.py](https://github.com/monty313/the-truth/blob/main/tests/test_ratchet_and_streak.py));
both Colab notebooks purge the counterfeit-era brains **by serial number**.

**Infection 3 — the forbidden classroom.** The trainer's own flea-jar lid ([THE_FLEA_CURE.md](THE_FLEA_CURE.md))
zero-weighted 59 of 90 days as "unwinnable" — the bot was *forbidden from ever practicing*
two-thirds of the world, then measured on all of it.
*Evidence:* the lid's removal (commit `853cfd4`: `w = torch.ones(D)` — every day
practices); the corrected map (**90/90 winnable**, quietest day +12.7% bound —
[evidence/the_70_sweep_2026-07-21.txt](evidence/the_70_sweep_2026-07-21.txt)).

**Infection 4 — the inherited flinch.** Warm-starts pointed at the proven-flat ancestor
first, so every new run began by copying the very brain the diseases had produced.
*Evidence:* warm chains reordered to the proven-PROFITABLE seed
(lift_best → PROVEN_LIFT, commit `86a2ae4`); known-flat serials purged from Drive at
notebook startup.

**The single root, named for the record:** the bot's issue was **INTENT DRIFT** — the gap
between what the owner meant (*"make x% a day, day after day, never breach"*) and what
the machine actually measured and paid. All four infections are instances of that one
disease. **The bot never had an issue of its own. It had ours, learned faithfully.** The
mirror was never cracked; it reflected a cracked scoreboard with perfect fidelity.

**Proof the cure took (measured, serial-stamped):** with the scoreboard healed and
nothing else — same body, same laws, same data — the same patient went from **1/90
cleared to 24/90, a 4-day row of banked +3%+ days** (Mar 18→23: +3.04/+3.19/+3.10/+3.13,
serial `c49091b393ca`, byte-verified), zero breaches throughout
([evidence/record_brain_scoreboard.txt](evidence/record_brain_scoreboard.txt)).

## C — CONCLUSION — and the bar against re-entry

The bot stands cured because its **feedback** stands cured. And this case cannot return
to this courtroom, because recurrence is barred by machinery, not promises:

1. **Pinned tests (36/36):** the streak law and THE LIFT ITSELF are permanent regression
   tests — if any engine, loader, or law drifts, the suite fails and names the day it broke.
2. **The Serial Rule:** no result exists without its own hash in its filename — counterfeit
   records are structurally impossible to confuse with real ones.
3. **The Antibody Law** ([THE_FLEA_CURE.md](THE_FLEA_CURE.md)): no impossibility claim
   without a measured bound — no training day may ever again be zero-weighted by belief.
4. **The one-door sweep:** decorative knobs deleted; cadence, brain size, and the owner's
   envelope read from single sources — intent now propagates everywhere or nowhere.
5. **The standing review gate:** the 4-agent team (correctness, durability, two-inputs,
   speed) reviews every new component — it is the very gate that caught Infection 2,
   which is the proof it works.
6. **World fingerprints + atomic saves + guarded loads:** records cannot silently cross
   into a changed world, and a crash cannot forge or destroy a champion.
7. **This precedence folder:** the disease is named, dissected, and shelved where every
   future session must find it before it repeats it.

**Held:** the bot's issue was intent drift in its feedback; the feedback is repaired,
measured, pinned, and guarded; the same charge cannot be brought again because the same
crime can no longer be silently committed. Counsel keeps his word to the court — you will
not see him here again for this.

*Filed as precedence. The mirror is clean. What we measure is now what we mean.* ⚖️
