# prime-rl → our Learn-to-Learn map

**Source:** [PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)  
**Local clone:** `data/knowledge/external/prime-rl`  
**What it is:** Async RL training at scale for LLMs (orchestrator + trainer + vLLM inference). **Not** a drop-in for our Channel1 gold-day policy — we **borrow intelligence patterns**, lab-legal only.

**Our skill:** learn LTF pullback vs continuation while HTF strong bull|bear on MARK SETS 1–4; keep child floor 35.

---

## What we will NOT do

| prime-rl | Why not wholesale |
|----------|-------------------|
| Multi-GPU FSDP / vLLM stack | We train a small MLP embryo on CPU/local, not 0.6B–1T LMs |
| Full GRPO on free rollouts | Mark soul + force-gate is teacher; unconstrained RL thrash kills floor |
| Replace KEEP/REJECT | Floor law is non-negotiable |

---

## Intelligence we *do* borrow (mapped)

### 1. Algorithm-blind loss components (rl · ce · ref_kl)

prime-rl splits training signal into **CE**, **RL advantage weights**, and **ref_kl** against a frozen teacher.

| prime-rl | Our lab |
|----------|---------|
| `ce` on teacher tokens | Mark act CE / path-class CE |
| `ref_kl` vs frozen model | **KL to child BEST** (`phaseA2` / pathmask KL) |
| `rl` advantages | Only after structure filter; never free reward crank |

**Rule:** when path structure is weak → **ref_kl / restore only** (or skip pack). When path strong → small CE on path-error acts + strong ref_kl.

### 2. Filters between rollout and training

prime-rl drops empty / zero-trainable / over-aged batches.

| prime-rl | Our lab |
|----------|---------|
| Filter zero-advantage batches | **`path_weak_skip_pack`** if path_class &lt; threshold |
| Drop over-off-policy rollouts | Each L2L round **reloads BEST state** before DAgger |
| Env verifiers | **KEEP/REJECT**: same≥35, breach=0, learn≠copy |

### 3. Async / off-policy awareness

prime-rl tracks how many policy versions contributed to a rollout (`max_off_policy_steps`, mismatch KL).

| prime-rl | Our lab |
|----------|---------|
| Cap off-policy staleness | DAgger on **current** policy path; never mix thrash weights into BEST without KEEP |
| Log mismatch KL | Log path_class vs act_match; reject if act high & path low (copying) |

### 4. On-policy distillation (OPD) idea

Policy samples, gradient from reverse KL to a **teacher** (not pure reward).

| prime-rl OPD | Our lab |
|--------------|---------|
| Teacher frozen LM | **Mark plan + structure laws** (HTF/LTF) + frozen child act |
| Reverse KL | Softmax KL act head → child embryo |

### 5. Group-relative credit (GRPO intuition)

Credit relative to a group, not absolute reward.

| prime-rl GRPO | Our lab |
|---------------|---------|
| Advantage vs group mean | Prefer **MWT miss days** as focus group; boost classes that dominate REJECT memory |
| Upweight hard examples (`max_rl`) | Higher sample weight on anti_thrash / miss_continuation / pullback |

### 6. Multi-turn trajectory merge

prime-rl merges multi-turn agent trajectories carefully (mask env tokens by default).

| prime-rl | Our lab |
|----------|---------|
| Multi-turn tools | **Day spine**: wait → pullback → continuation → bank |
| Mask observation tokens | Don't BC full day memo; label **path laws** on decision bars |

### 7. Environments + verifiers

prime-rl uses verifiable env rewards.

| prime-rl | Our lab |
|----------|---------|
| Env success | **practice same / mwt / breach** dual scores |
| Forward eval separate | Held-out / forward scripts; never train on exam window |

---

## Concrete lab recipe (prime-rl informed)

```
1. Rollout  = DAgger walk under mark_doctrine (all Official Sets)
2. Filter   = structure path_class quality; skip pack if weak
3. Signal   = path CE on struct features (pullback|continuation|htf_strong)
              + optional act CE only on path-error classes
              + ref_kl act → child BEST
4. Update   = freeze trunk; train path head (struct-aug); surgical act
5. Verify   = KEEP only same≥35 breach=0 learn≠copy + improvement
6. Memory   = boost failed path laws next round (meta)
```

---

## Priority for our climb (past 35)

1. **Struct-aug path head** (HTF/LTF features) — already in flight  
2. **ref_kl floor protect** — already law  
3. **Filters** — path_weak_skip_pack (KEEP)  
4. **Later ADD:** group-relative day weights (hard MWT days like max_rl)  
5. **Later ADD:** log mismatch metrics (act_match vs path_class) to KAG each round  

---

## Files

- Clone: `01_SYSTEM/data/knowledge/external/prime-rl`  
- This map: `data/knowledge/army/PRIME_RL_L2L_MAP.md`  
- Upstream: https://github.com/PrimeIntellect-ai/prime-rl  
