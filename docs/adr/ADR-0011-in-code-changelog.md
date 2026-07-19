# ADR-0011: Every change carries its WHY in the code (Monty's standing rule, 2026-07-19)

**Monty's instruction (verbatim intent):** "from now on make comments in the code
for WHY you updated something, and leave instructions to keep doing that on the
next time something is updated — that way we never get lost."

**Decision (permanent, applies to EVERY file from now on):**
1. Every source file carries a **CHANGE LOG** block at the end of its 5W+I header.
2. On ANY edit to a file, the editor APPENDS one line at the top of that block:
   `- YYYY-MM-DD  <short what>  — WHY: <reason>`
   (newest first). Never delete old lines — the log is the memory.
3. The block ENDS with a standing instruction line that must be KEPT:
   `# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.`
4. Non-obvious inline changes also get a short `# why:` comment at the point of change.
5. `scripts/check_changelog.py` reports any source file missing the block — run it
   before committing; it is the gentle guard that keeps this honest.

**Why this matters:** a project this size drifts if the reasoning lives only in chat
or commit messages. Putting the WHY next to the code means the next session (human or
LLM) reconstructs intent by reading the file, not by archaeology. This complements the
5W+I header (what/why the file exists) with a running WHY-it-changed history.
