# Momentum One — simple commands
# Usage: make prove | make train | make lint | make test

ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
export PYTHONPATH := $(ROOT)

.PHONY: help prove train preflight lint test backtest live hud

help:
	@echo Targets:
	@echo   make prove      - score champion at 3.0 / 3.5
	@echo   make train      - GPU train
	@echo   make preflight  - sanity before train
	@echo   make lint       - ruff check
	@echo   make test       - pytest
	@echo   make backtest   - sample signal backtest
	@echo   make live       - MT5 bridge
	@echo   make hud        - local HUD server

prove:
	python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5

train:
	python scripts/gpu_train.py

preflight:
	python scripts/preflight_train.py

lint:
	python -m ruff check training features signals scripts core || echo "install ruff: pip install ruff"

test:
	python -m pytest tests -q

backtest:
	python scripts/backtest_momentum_vector.py

live:
	python scripts/run_live.py --days 1

hud:
	python scripts/run_hud.py
