SIGON one-zip install
=====================
Extract this archive INTO the-truth repo root (overwrite when asked).

Paths inside match the repo:
  scripts/gpu_train.py
  training/gpu_data.py
  training/day_board.py
  training/signal_accuracy.py
  configs/features.yaml  (include_signal_agent_slots: true)
  configs/goals.yaml
  configs/data.yaml
  configs/sigon_train.yaml
  configs/signal_accuracy_schema.yaml
  evaluation/consistency.py
  hud/iron_man_sigon.html
  doctrine/cmo_inbox/
  COLAB_SIGON_RUN.md

After extract (Windows PowerShell):
  cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
  del artifacts\gpu_cache_*.npz
  if (Test-Path artifacts\symbol_cache) { Remove-Item -Recurse -Force artifacts\symbol_cache }

NEVER delete .pt checkpoints when clearing caches.

Train (Colab L4 or local GPU):
  python scripts/gpu_train.py --csv-dir data --instances 8000 --minutes 600

OOM: --instances 4000 then 2000 then 1024
Champion: artifacts/checkpoints/best_sigon.pt
obs_dim ~6820 with signals ON — do NOT load PROVEN 1820 weights.
