# DAILY SCRIPTS (only these matter most days)

**Look at this file first when opening `scripts/`.**  
There are many scripts. Most are rare. These are the daily ones.

---

## Daily (goal loop)

| Script | Job |
|--------|-----|
| **prove_it.py** | **THE score** — clear % / breach % |
| **preflight_train.py** | Ready check |
| **self_heal_epoch.py** | Diagnose → dials → gate |
| **give_llm_what_it_needs.py** | IRAC / mind pack for AI |
| **consistency_sprint.py** | GPU climb clear rate |
| **meta_train.py** | Meta-tuner rewards/hparams |
| **restore_meta_tuner.py** | Repair meta_tuner if broken |

### Commands

```text
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
python scripts/preflight_train.py
python scripts/self_heal_epoch.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 12
python scripts/consistency_sprint.py --minutes 600 --envs 256
python scripts/meta_train.py --minutes 600
```

Or use **`USE/`** buttons (no typing).

---

## Sometimes

| Script | Job |
|--------|-----|
| mind_probe_day.py | One-day policy MRI |
| diagnose_day.py | Day diagnosis |
| run_live.py | MT5 bridge |
| run_hud.py | HUD |
| gpu_train.py | Alternate train entry |
| replay_best.py | Replay a checkpoint |

---

## Rare / research (ignore unless asked)

`backtest_*`, `score_*`, `improve_dvmr*`, `optimize_*`, `verify_*`, `_*.py`

---

## Rule for AI

If you add a script Monty will run **often**:

1. List it in **this file** (top table)  
2. Add a **`USE/N_name.bat`** button  
3. Mention it in **DO_THIS.md**
