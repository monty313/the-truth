# Colab L4 SIGON + Jarvis

1. Runtime → L4 GPU
2. Mount Drive, clone/pull the-truth, put 4 CSVs in data/
3. rm caches once: artifacts/gpu_cache_*.npz artifacts/symbol_cache
4. TRAIN cell: `python scripts/gpu_train.py --csv-dir data --instances 8000 --minutes 600`
5. JARVIS cell (anytime): `python scripts/jarvis_talk.py status` / `board` / `SET w_pullback_with_htf=0.35` / `RELOAD_REWARDS`

Auto-save: best_sigon.pt | 3 fails same day → that instance new day | hot updates no restart
