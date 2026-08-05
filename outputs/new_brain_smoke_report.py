"""One-shot smoke report for adaptive_rl_brain_7_31_26 (sandbox only)."""
from __future__ import annotations

from collections import Counter

from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
)
from lineages.adaptive_rl_brain_7_31_26.train_stub import run_day_rollout, train_stub_epoch


def main() -> None:
    print("=== PRACTICE DAY (greedy tiny policy) ===")
    r = run_day_rollout(max_steps=30, greedy=True, seed=42, decide_every=20)
    tags = Counter(s.tag.value for s in r.steps)
    acts = Counter(s.action for s in r.steps)
    rews = [s.reward for s in r.steps]
    pos = sum(1 for x in rews if x > 0)
    neg = sum(1 for x in rews if x < 0)
    zero = sum(1 for x in rews if x == 0)
    print(f"steps: {r.n_steps}")
    print(f"total_reward: {r.total_reward:.4f}")
    print(f"mean_reward: {r.total_reward / max(r.n_steps, 1):.4f}")
    print(f"reward_sign: +{pos}  0={zero}  -{neg}")
    print(f"tags: {dict(tags)}")
    print(
        "actions: "
        f"hold={acts.get(ACTION_HOLD, 0)} "
        f"buy={acts.get(ACTION_BUY, 0)} "
        f"sell={acts.get(ACTION_SELL, 0)}"
    )
    print("last_5:")
    for s in r.steps[-5:]:
        print(f"  t={s.t} act={s.action} tag={s.tag.value} rew={s.reward:.3f}")

    print()
    print("=== SHORT TRAIN SMOKE (REINFORCE) ===")
    for seed in (1, 2, 3):
        out = train_stub_epoch(steps=15, seed=seed)
        print(
            f"seed={seed} n={int(out['n'])} "
            f"mean_reward={out['mean_reward']:.4f} loss={out['loss']:.4f}"
        )

    print()
    print("=== NOTE ===")
    print("SANDBOX smoke on synthetic days.")
    print("Not PROVEN prove_it. Not live money. Not clear%/breach%.")


if __name__ == "__main__":
    main()
