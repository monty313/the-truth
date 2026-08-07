"""Bridge: the-truth Teacher 1 (trade) partners with ARMY Reason Teacher (Teacher 2).

Additive only. Does not rewrite perception/sets or PROVEN.
When ARMY markos_core is importable, calls reason_one_pack on full-obs vectors.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# Optional ARMY path
_ARMY_CORE = Path(r"C:\Users\user\OneDrive\Desktop\ARMY\01_SYSTEM\packages\core")
if _ARMY_CORE.is_dir() and str(_ARMY_CORE) not in sys.path:
    sys.path.insert(0, str(_ARMY_CORE))


def reason_with_army(
    obs_vector: Sequence[float] | None = None,
    *,
    obs_dict: Mapping[str, Any] | None = None,
    day: str = "",
    bar_index: int = 0,
) -> dict[str, Any]:
    """Teacher 2 reason pack; falls back to local teach_one_bar if ARMY missing."""
    try:
        from markos_core.kag_mark.reason_teacher import reason_one_pack

        return reason_one_pack(
            obs_vector,
            obs_dict=obs_dict,
            day=day,
            bar_index=bar_index,
            persist=True,
            partner_with_trade_teacher=True,
            write_outputs=True,
        )
    except Exception as e:
        from lineages.adaptive_rl_brain_7_31_26.kag_teachers.teachers import teach_one_bar

        trade = teach_one_bar(obs_dict)
        return {
            "ok": False,
            "fallback": "trade_teacher_only",
            "error": str(e),
            "trade_teacher": trade,
            "note": "Install/import ARMY markos_core for full Reason Teacher (Teacher 2)",
        }


def partner_cycle() -> dict[str, Any]:
    try:
        from markos_core.kag_mark.reason_teacher import run_reason_teacher_cycle

        return run_reason_teacher_cycle(write=True)
    except Exception as e:
        return {"ok": False, "error": str(e)}
