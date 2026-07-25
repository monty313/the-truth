#!/usr/bin/env python3
"""Manage the 500 observation signal slots.

  python scripts/manage_signal_slots.py summary
  python scripts/manage_signal_slots.py list [--family camillion|momentum_one]
  python scripts/manage_signal_slots.py next-free
  python scripts/manage_signal_slots.py kinds
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from signals.encode import (  # noqa: E402
    N_SLOTS,
    KIND_HANDLERS,
    load_slot_config,
    load_filled_slots,
    list_kinds,
)


def cmd_list(family: str | None) -> int:
    cfg = load_slot_config()
    filled = cfg.get("filled") or {}
    rows = []
    for k, v in sorted(filled.items(), key=lambda x: int(x[0])):
        if family and v.get("family") != family:
            continue
        en = "ON " if v.get("enabled", True) else "off"
        rows.append(
            f"  {int(k):3d}  {en}  {v.get('name', '?'):40s}  kind={v.get('kind')}  "
            f"family={v.get('family', '?')}  fid={v.get('fidelity', 'native')}"
        )
    print(f"Filled definitions: {len(filled)}  (showing {len(rows)})")
    print("\n".join(rows) if rows else "  (none)")
    return 0


def cmd_next_free() -> int:
    filled = load_slot_config().get("filled") or {}
    used = {int(k) for k in filled}
    for i in range(N_SLOTS):
        if i not in used:
            print(i)
            return 0
    print("none", file=sys.stderr)
    return 1


def cmd_summary() -> int:
    cfg = load_slot_config()
    filled = cfg.get("filled") or {}
    enabled = load_filled_slots(only_enabled=True)
    by_fam: dict[str, int] = {}
    for v in filled.values():
        fam = v.get("family") or "unknown"
        by_fam[fam] = by_fam.get(fam, 0) + 1
    print(f"n_slots={N_SLOTS}")
    print(f"defined={len(filled)}  enabled={len(enabled)}  free={N_SLOTS - len(filled)}")
    print("by family:")
    for fam, n in sorted(by_fam.items()):
        print(f"  {fam}: {n}")
    print(f"kinds available: {len(KIND_HANDLERS)}")
    return 0


def cmd_kinds() -> int:
    for k in list_kinds():
        print(k)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list")
    p_list.add_argument("--family", default=None)
    sub.add_parser("next-free")
    sub.add_parser("summary")
    sub.add_parser("kinds")
    args = ap.parse_args()
    if args.cmd == "list":
        return cmd_list(args.family)
    if args.cmd == "next-free":
        return cmd_next_free()
    if args.cmd == "summary":
        return cmd_summary()
    if args.cmd == "kinds":
        return cmd_kinds()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
