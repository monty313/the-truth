"""Run cards — the Memory (MLflow-optional experiment tracker).

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty (ADR-0008 metadata standard).
WHAT:  Creates one run card per run: full metadata standard, metrics,
       artifacts list, summary. JSON on disk always; mirrored to MLflow
       when the package is installed (laptop-optional).
WHEN:  2026-07-19 overnight build.
WHERE: artifacts/runs/<run_id>/run_card.json ; used by training,
       gauntlet, evaluation; spans join via run_id (telemetry/tracer).
WHY:   Every run must be comparable and auditable forever; champion-vs-
       challenger needs honest lineage.
INTERCONNECTED WITH: telemetry/tracer.set_run, configs/* (hashed in),
       evaluation/champion.py (reads cards), HUD (shows latest card).
----------------------------------------------------------------------

CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  created/last-major  — WHY: v0.1 build + v0.2 audit fixes (see docs/AUDIT_FIXES_2026-07-19.md).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
from __future__ import annotations
import hashlib, json, os, subprocess, time, uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "artifacts", "runs")
os.makedirs(RUNS, exist_ok=True)

try:  # optional mirror
    import mlflow  # type: ignore
    _MLF = True
except Exception:
    _MLF = False


def _cfg_hash() -> str:
    h = hashlib.sha256()
    cfg_dir = os.path.join(ROOT, "configs")
    for fn in sorted(os.listdir(cfg_dir)):
        with open(os.path.join(cfg_dir, fn), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def _code_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


class Run:
    """One experiment run. Everything the metadata standard demands."""

    def __init__(self, kind: str, *, symbols, timeframes, data_window,
                 seed: int, assumptions: dict, parent_run: str | None = None):
        self.id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        self.dir = os.path.join(RUNS, self.id)
        os.makedirs(self.dir, exist_ok=True)
        self.card = {
            "run_id": self.id, "kind": kind, "started": time.time(),
            "code_version": _code_version(), "config_version": _cfg_hash(),
            "reward_version": _cfg_hash(),  # rewards.yaml folded into config hash
            "mask_version": "forever_masks_v1", "registry_version": "codex_v1",
            "model_version": None, "checkpoint_lineage": [parent_run] if parent_run else [],
            "data_window": data_window, "symbols": symbols, "timeframes": timeframes,
            "seed": seed, "assumptions": assumptions,   # broker + execution (fills!)
            "metrics": {}, "artifacts": [], "summary": None,
        }
        from telemetry import tracer
        tracer.set_run(self.id)
        if _MLF:
            mlflow.set_experiment("momentum_one")
            mlflow.start_run(run_name=f"{kind}_{self.id}")
            mlflow.log_params({k: str(v)[:200] for k, v in self.card.items()
                               if k in ("kind", "code_version", "config_version",
                                        "symbols", "seed")})

    def log(self, **metrics) -> None:
        self.card["metrics"].update(metrics)
        if _MLF:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float))})

    def artifact(self, path: str) -> None:
        self.card["artifacts"].append(path)
        if _MLF:
            try: mlflow.log_artifact(path)
            except Exception: pass

    def finish(self, summary: str, model_version: str | None = None) -> str:
        self.card["summary"] = summary
        self.card["model_version"] = model_version
        self.card["ended"] = time.time()
        p = os.path.join(self.dir, "run_card.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.card, f, indent=2, default=str)
        if _MLF:
            mlflow.end_run()
        return p
