"""Talk while training. Second Colab cell.
  python scripts/jarvis_talk.py status|board|outbox
  python scripts/jarvis_talk.py "SET w_pullback_with_htf=0.35"
  python scripts/jarvis_talk.py "RELOAD_REWARDS"
  python scripts/jarvis_talk.py "NOTE dual HTF — take LTF pulls"
"""
from __future__ import annotations
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.configs import path as rpath

def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__); return
    cmd = args[0]
    jarvis = rpath("artifacts", "jarvis"); os.makedirs(jarvis, exist_ok=True)
    if cmd == "status":
        p = os.path.join(jarvis, "status.json")
        print(open(p, encoding="utf-8").read() if os.path.exists(p) else "No status yet")
        return
    if cmd == "board":
        p = rpath("artifacts", "llm_curriculum", "day_board.json")
        if not os.path.exists(p):
            print("No day_board yet"); return
        j = json.load(open(p, encoding="utf-8"))
        print("clear", j.get("clear_rate"), "breach", j.get("breach_rate"), "row", j.get("row"))
        for d in (j.get("days") or [])[:40]:
            print(d.get("emoji"), d.get("status"), d.get("pnl"), d.get("goal"), d.get("symbol"))
        return
    if cmd == "outbox":
        p = os.path.join(jarvis, "outbox.md")
        print(open(p, encoding="utf-8").read() if os.path.exists(p) else "(empty)")
        return
    msg = " ".join(args)
    with open(os.path.join(jarvis, "inbox.md"), "a", encoding="utf-8") as f:
        f.write(msg.strip() + "\n")
    print("Jarvis inbox ←", msg.strip())

if __name__ == "__main__":
    main()
