# ADR-0007: META-LEARNING RL — final
Monty's final call 2026-07-18 (Claude recommended meta-lite; overruled). Implementation =
staged: context-based meta (recurrent, goal-conditioned, task-randomized) first; MAML-style
only if evidence demands + budget allows. Meta-optimizer proposes reward-weight/hparam
configs; NEVER auto-adopts; Monty approves each adoption. Budget ~$50 cloud bursts.
Hardware: Dell Latitude 7280, i5-6300U 2c/4t, 8GB, no GPU.
