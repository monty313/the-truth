# Learn-to-learn path (student meta) — HTF + LTF skill

## Timeframe sets (MARK SETS LAW — remember these)

**Immutable** (`perception/sets.py` · `MARK_SETS_LAW`).  
**LTF = first TF** (pullback / continuation / add).  
**HTF = last two TFs** (trend confirm — strong bull or bear).  
**Scan all four** under `mark_doctrine` (never Set-2-only for Mark path).

| Set | Name | LTF (entry) | HTF (confirm) | Stack |
|----:|------|-------------|---------------|-------|
| **1** | micro | **1m** | **15m, 30m** | 1m \| 15m, 30m |
| **2** | intraday | **5m** | **30m, 1h** | 5m \| 30m, 1h |
| **3** | swing | **15m** | **1h, 4h** | 15m \| 1h, 4h |
| **4** | macro | **30m** | **4h, 1d** | 30m \| 4h, 1d |

Sub-sets (weaker): A 1m→5m · B 5m→15m · C 15m→30m · D 1h→4h · E 4h→1d.

## Correct skill

The bot must **learn to learn** to:

1. Detect **HTF strong bull or strong bear** on the set’s confirmation stack (last two TFs).
2. On that set’s **LTF entry TF**, identify:
   - **Pullback** — LTF opposes HTF → wait / HOLD (slingshot load).
   - **Continuation** — LTF aligns with HTF → fire / add with the tide.
3. If HTF is **not** strong on the active set → do not invent trades.
4. Never thrash on pullbacks; never miss continuations with HTF.
5. Prefer multi-set Mark opportunity (`mark_opportunity.best`) so any of sets 1–4 can teach the same law.

This is calendar-free structure — not day memos.

## Path laws

| Law | Meaning |
|-----|---------|
| `ltf_pullback_htf_strong` | HTF clear + LTF opposite → pullback wait |
| `ltf_continuation_htf_strong` | HTF clear + LTF same → continuation fire |
| `htf_not_strong` | No clear HTF tide → flat |
| `anti_thrash` | Fired when should wait |
| `miss_continuation` | Held when should continue with HTF |
| `hold_on_spine` | Award protect HOLD |

Labels use live `perceive()` higher/lower + structure.pullback (same as `structure.py`).

## Loop

1. DAgger on policy path → Mark act + HTF/LTF path law  
2. Phase A: train path-law head (structure ID)  
3. Phase B: surgical act only on path-error laws + high KL (if path quality OK)  
4. learn≠copy + KEEP/REJECT on practice pack  
5. Pattern memory boosts failing laws next round  

## Files

- `learn_to_learn_path.py`  
- `L2L_JOINT_CONSISTENCY__charter.md`  
- `L2L_PATH_MEMORY.jsonl` · `L2L_PATH_STATE__latest.json`  

## prime-rl intelligence (mapped)
See PRIME_RL_L2L_MAP.md � borrow filters, ref_kl, off-policy hygiene; do not import GPU stack.

