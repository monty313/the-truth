# Run the bot's training on your PC (continuous)

This trains the bot **non-stop** — no 10-minute cloud limit. It keeps the most
**consistent** policy it finds and never overwrites it with a worse one.

Everything is already in this folder:
`Fable5_Foundation\MOMENTUM_ONE\01_BOT_CODE\momentum_one\`

---

## Step 1 — one time only: install Python + the libraries

1. If you don't have Python 3.10, install it (the installer is in
   `Downloads\_INSTALLERS\python-3.10.0-amd64.exe`). **Tick "Add Python to PATH."**
2. Open **Command Prompt**, then paste:

   ```
   cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\01_BOT_CODE\momentum_one
   pip install numpy pandas pyyaml torch
   ```

   (`torch` is a ~200 MB download — one time.)

---

## Step 2 — start training

Double-click **`RUN_ON_PC.bat`** in this folder.

That's it. A black window opens and it trains continuously, resuming from where
it left off each time. Leave it running overnight.

---

## What's happening

- It drills **one week** (Jan 27 – Feb 2, an average week) over and over.
- The target: **+5%/day (2× your goal) with no −4% breach.**
- It saves the best policy to **`artifacts\checkpoints\best_trading.pt`**,
  and only replaces it when a **more consistent** policy appears.

## Check progress anytime

Open **`artifacts\drill2x_progress.json`**. Look at:
- `days_at_2x` — how many of the 5 days hit 2× (goal = 5/5)
- `days_at_goal` — how many hit at least 1× the goal
- `breaches` — must stay **0**
- `best_consistency_2x` — the best it has locked in

## Stop / resume

- **Stop:** close the black window (or Ctrl+C). Progress is already saved.
- **Resume:** double-click `RUN_ON_PC.bat` again — it picks up the checkpoint.

## The standard it's aiming for
See **`THE_TARGET_oracle_benchmark.md`** — the proof there's real edge to reach.
