"""SHA-256 helpers for experiment identity pins.

CHANGE LOG:
- 2026-07-31  honest gate — WHY: freeze checkpoint/dials/data/meaning before any
  training cycle so results are reproducible and not silent-edit contaminated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def file_sha256(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    """Stable UTF-8 JSON for hashing (sorted keys, no whitespace variance)."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_bytes(canonical_json_bytes(obj))


def pin_paths(paths: Mapping[str, str | Path]) -> dict[str, dict[str, Any]]:
    """Return {name: {path, exists, size, sha256}} for each path."""
    out: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        p = Path(path)
        if not p.is_file():
            out[name] = {
                "path": str(p).replace("\\", "/"),
                "exists": False,
                "size": 0,
                "sha256": None,
            }
            continue
        out[name] = {
            "path": str(p).replace("\\", "/"),
            "exists": True,
            "size": int(p.stat().st_size),
            "sha256": file_sha256(p),
        }
    return out
