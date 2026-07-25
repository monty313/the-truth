# doctrine/

## Start here for the LLM
| File | What |
|------|------|
| **SYSTEM_DOCTRINE_CMO.md** | **Full CMO + Lead Quant persona (RAG SSOT)** |
| LLM_JOB.md | Short job card |
| LLM_THINKS_LIKE_MONTY.md | Build-on principles |
| policy_skill.md | SkillOpt memory (gated edits) |
| LLM_REGIME_DEFINITIONS.yaml | Regime + indicator registry |
| STANDING_LAWS.md | Hard laws |

## How to update
- Persona changes → edit **SYSTEM_DOCTRINE_CMO.md** only, then sync the short cards if needed.
- Regimes/indicators → **LLM_REGIME_DEFINITIONS.yaml** only.
- Skill memory → gated via `scripts/skillopt_gate.py`.
- Do not delete PERFORMANCE_IS_POSSIBLE* files at repo root.
