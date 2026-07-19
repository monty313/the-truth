"""Generate meta-optimizer PROPOSALS for Monty to approve (ADR-0007).
5W+I: WHO Claude; adoption ONLY by Monty. WHAT writes a proposal batch of
reward-weight/hparam configs to artifacts/proposals/. WHEN 2026-07-19 (audit
R8: propose() was never called). WHY the self-optimization pillar must be
reachable. INTERCONNECTED: training/meta_optimizer, rewards/training configs.
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from training.rewards import load_weights          # noqa: E402
from training.meta_optimizer import propose         # noqa: E402
from core.configs import training_cfg               # noqa: E402

if __name__ == "__main__":
    hp = dict(training_cfg().get("ppo", {}))
    path = propose(load_weights(), hp, n=8, seed=1)
    print("Proposals written:", path)
    print("These are PROPOSALS ONLY — review, then adopt with a new ADR (ADR-0007).")
    print(json.dumps(json.load(open(path))["trials"][0], indent=2)[:600])
