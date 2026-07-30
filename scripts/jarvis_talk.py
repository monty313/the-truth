"""Talk to training like Jarvis — run in a SECOND Colab cell while gpu_train runs.

Usage:
  python scripts/jarvis_talk.py status
  python scripts/jarvis_talk.py board
  python scripts/jarvis_talk.py "NOTE push pullbacks harder under dual HTF"
  python scripts/jarvis_talk.py "SET w_pullback_with_htf=0.35"
  python scripts/jarvis_talk.py "RELOAD_REWARDS"
"""
from __future__ import annotations
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT)

from core.configs import path as rpath


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd = args[0]
    jarvis = rpath("artifacts", "jarvis")
    os.makedirs(jarvis, exist_ok=True)

    if cmd == "status":
        p = os.path.join(jarvis, "status.json")
        if not os.path.exists(p):
            print("No status yet — is gpu_train running?")
            return
        print(open(p, encoding="utf-8").read())
        return

    if cmd == "board":
        p = rpath("artifacts", "llm_curriculum", "day_board.json")
        if not os.path.exists(p):
            print("No day_board yet")
            return
        j = json.load(open(p, encoding="utf-8"))
        print("clear_rate", j.get("clear_rate"), "breach", j.get("breach_rate"), "row", j.get("row"),
              "obs_dim", j.get("obs_dim"))
        rows = j.get("rows") or j.get("days") or []
        for d in rows[:40]:
            print(d.get("emoji"), d.get("status"), "pnl", d.get("pnl"),
                  "goal", d.get("target", d.get("goal")), d.get("symbol"))
        return

    if cmd == "outbox":
        p = os.path.join(jarvis, "outbox.md")
        print(open(p, encoding="utf-8").read() if os.path.exists(p) else "(empty)")
        return

    # anything else = message into inbox
    msg = " ".join(args)
    inbox = os.path.join(jarvis, "inbox.md")
    with open(inbox, "a", encoding="utf-8") as f:
        f.write(msg.strip() + "\n")
    print("Jarvis inbox ←", msg.strip())
    print("Train loop will apply within a few updates (hot, no restart).")


if __name__ == "__main__":
    main()
