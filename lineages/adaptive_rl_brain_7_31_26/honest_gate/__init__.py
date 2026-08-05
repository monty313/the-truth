"""Honest pre-training gate for multi-pair tutor (practice-only, shell-locked).

Lineage only. Never touches models/PROVEN_*.pt.
"""
from __future__ import annotations

__all__ = [
    "file_sha256",
    "build_meaning_manifest",
    "meaning_hash",
    "assert_meaning_matches_frozen",
    "verify_shell_locked",
    "assert_no_day_leak",
    "build_data_contract",
]
