"""Config-door proof (LAWS #3). 5W+I: see test_shell.py header.
CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  created/last-major  — WHY: v0.1 build + v0.2 audit fixes (see docs/AUDIT_FIXES_2026-07-19.md).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.configs import load, path


def test_every_config_loads():
    for f in glob.glob(path("configs", "*.yaml")):
        name = os.path.splitext(os.path.basename(f))[0]
        assert isinstance(load(name), dict)


def test_no_ruled_number_hardcoded_in_simulator():
    """The Shell constants must originate from configs, not literals."""
    import backtesting.simulator as S
    from core.configs import shell_cfg
    sc = shell_cfg()
    assert S.ATR_STOP_MULT == sc["broker_stop"]["atr_mult"]
    assert S.POINT_SIZE == sc["point_size"]
