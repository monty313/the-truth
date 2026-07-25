SIGON pack — extract INTO your the-truth repo root so paths match:

  scripts/gpu_train.py
  training/gpu_data.py
  training/day_board.py
  training/signal_accuracy.py
  configs/features.yaml   (signals ON)
  configs/goals.yaml
  configs/data.yaml
  configs/sigon_train.yaml
  configs/signal_accuracy_schema.yaml
  evaluation/consistency.py
  hud/iron_man_sigon.html
  doctrine/cmo_inbox/
  COLAB_SIGON_RUN.md

Then:
  del artifacts\gpu_cache_*.npz
  (remove artifacts\symbol_cache if present)

Colab / train:
  python scripts/gpu_train.py --csv-dir data --instances 8000 --minutes 600

Put 4 M1 CSVs in data/: XAUUSD, EURUSD, GBPUSD, US30
