# Overnight handoff — Spine Shadow

**Status when Mark sleeps:** safe climb loop running.  
**Read first:** `WHAT_WORKS__GOAL.md` (only durable wins for the goal).

## Board

| Meter | Value |
|-------|------:|
| Practice best same | **35**/50 |
| mwt | 15 |
| breach | 0 |
| Oracle spines | **50**/50 breach 0 |
| PROVEN | untouched (see PROVEN_FINGERPRINT__session.md) |

## Running now

```text
python lineages/adaptive_rl_brain_7_31_26/spine_safe_one_day.py --max-rounds 36
```

Log: scratch `spine_safe_one_day.log` · cycles `SPINE_SAFE_ONE_DAY__latest.json`  
On each KEEP: appends to `WHAT_WORKS__GOAL.md` + updates `BEST__latest.json` + `mark_clone_full_obs_v1.pt`.

## If you wake and climb is done

1. `BEST__latest.json` — new same?  
2. `WHAT_WORKS__GOAL.md` — any KEEP rows added?  
3. Forward:  
   `python lineages/adaptive_rl_brain_7_31_26/score_forward_100d.py --n-days 100`  
   (uses `data/raw/XAUUSD_M1_full.csv`, 100 days never in fit set)

## Goal finish line (unchanged)

Dual held-out score, **same = n_days**, **breach = 0**, empty fit intersection — not practice-only celebration.

## Do not

- Open 3 teachers  
- Crank entry rewards as primary  
- Save when pack collapses  
- Touch PROVEN  
