#!/usr/bin/env python3
"""Compare Drive Google Sheet content to local student CSV."""
from __future__ import annotations

import csv
import io
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

MCP = Path(
    r"C:\Users\user\.grok\sessions\C%3A%5CUsers%5Cuser%5CFable5_Foundation%5CMOMENTUM_ONE%5Cthe-truth"
    r"\019fc0db-db44-7d93-88c6-c015dfca2409\mcp\call-e189ce53-1e65-4979-884f-68e14f09a315-34.json"
)
LOCAL = Path("Student Strategy Tests.xlsx - Student Results.csv")


def main() -> int:
    obj = json.loads(MCP.read_text(encoding="utf-8"))
    content = obj.get("content") or ""
    print("content_len", len(content))
    print("truncated", obj.get("truncated"))
    print("sheets", re.findall(r"--- Sheet: ([^-]+) ---", content))
    print("sheet markers", content.count("--- Sheet:"))

    lines = content.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Student Name,"))
    body = "\n".join(lines[start:])
    if "--- Sheet:" in body:
        body = body.split("--- Sheet:")[0]
    drive_rows = list(csv.DictReader(io.StringIO(body)))
    print("drive_n", len(drive_rows), "cols", list(drive_rows[0].keys()) if drive_rows else None)

    local_rows = []
    with LOCAL.open(encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        for i, row in enumerate(r):
            if i < 3:
                continue
            if not row or row[0] == "Student Name" or not row[0]:
                continue
            if len(row) < 6:
                continue
            local_rows.append(tuple(row[:7]))

    drive_tuples = []
    for r in drive_rows:
        drive_tuples.append(
            (
                r.get("Student Name", ""),
                r.get("Date Added to this List", ""),
                r.get("Start Date", ""),
                r.get("Strategy Tested", ""),
                r.get("Win Rate", ""),
                r.get("Total Return", ""),
                r.get("Backtesting Project Link", ""),
            )
        )

    print("local_n", len(local_rows))
    set_d = set(drive_tuples)
    set_l = set(local_rows)
    print("only_in_drive", len(set_d - set_l))
    print("only_in_local", len(set_l - set_d))
    print("intersection", len(set_d & set_l))

    by = defaultdict(list)
    for t in drive_tuples:
        s = t[3].strip()
        if not s:
            continue
        try:
            wr = float(str(t[4]).replace("%", ""))
        except ValueError:
            wr = None
        try:
            ret = float(str(t[5]).replace("%", ""))
        except ValueError:
            ret = None
        by[s].append((wr, ret))

    print()
    print("Strategy | n | WRmed | RETmed | all_ret>0")
    for s, xs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        wrs = [w for w, r in xs if w is not None]
        rets = [r for w, r in xs if r is not None]
        print(
            s,
            len(xs),
            round(st.median(wrs), 1) if wrs else None,
            round(st.median(rets), 1) if rets else None,
            all(r > 0 for r in rets) if rets else None,
        )

    # keyword scan for rule detail
    low = content.lower()
    for kw in [
        "rule",
        "entry",
        "exit",
        "definition",
        "timeframe",
        "drawdown",
        "trade count",
        "# trades",
        "r-multiple",
        "stop loss",
        "take profit",
        "indicator",
        "pine",
    ]:
        print("kw", kw, "->", kw in low)

    # show if more sheets after first
    parts = re.split(r"--- Sheet: ", content)
    print("parts after split", len(parts))
    for p in parts[1:]:
        name = p.split(" ---", 1)[0]
        print("sheet part name:", name, "chars", len(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
