"""Guard: every source file must carry a CHANGE LOG block (ADR-0011).
5W+I: WHO Claude for Monty. WHAT lists source files missing the in-code
CHANGE LOG block so we never lose the WHY of a change. WHEN 2026-07-19.
WHERE run before committing. WHY Monty's standing rule 'we never get lost'.
INTERCONNECTED: ADR-0011, docs/LAWS.md #9, every *.py.

CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  created  — WHY: enforce Monty's standing in-code-changelog rule (ADR-0011).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = ("__pycache__", ".git", "artifacts", "logs", ".venv")
missing = []
for dp, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in SKIP]
    for f in files:
        if f.endswith(".py") and f != "__init__.py":
            txt = open(os.path.join(dp, f), encoding="utf-8").read()[:3000]
            if "CHANGE LOG" not in txt:
                missing.append(os.path.relpath(os.path.join(dp, f), ROOT))
if missing:
    print(f"MISSING CHANGE LOG block in {len(missing)} file(s) (ADR-0011):")
    for m in sorted(missing):
        print("  -", m)
    sys.exit(1)
print("OK — every source file carries a CHANGE LOG block (ADR-0011).")
