"""Config-door proof (LAWS #3). 5W+I: see test_shell.py header."""
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
