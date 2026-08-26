# Divide-and-Collapse — task runner.
# Works under git-bash on Windows and any POSIX shell; PowerShell users can
# call the underlying commands directly.

.PHONY: help install test test-slow lint format example data judgelight figs clean

help:
	@echo "Targets:"
	@echo "  install     Install package + dev/viz extras into a uv venv"
	@echo "  test        Run fast tests (pytest -m 'not slow')"
	@echo "  test-slow   Run the full suite including stress tests"
	@echo "  lint        ruff + black --check + mypy"
	@echo "  format      ruff --fix + black"
	@echo "  example     Run the paper's running example end to end"
	@echo "  data        Regenerate the POGEMA-derived input schedules (long)"
	@echo "  judgelight  Fetch Tang et al.'s Judgelight baseline (pinned SHAs)"
	@echo "  figs        Regenerate the paper's figures from results/raw/"
	@echo "  clean       Remove build/cache artefacts"

install:
	uv venv
	uv pip install -e ".[dev,viz]"

test:
	uv run pytest -m "not slow"

test-slow:
	uv run pytest

lint:
	uv run ruff check src tests experiments examples
	uv run black --check src tests experiments examples
	uv run mypy

format:
	uv run ruff check --fix src tests experiments examples
	uv run black src tests experiments examples

example:
	uv run python examples/running_example.py

data:
	uv run python scripts/dump_pogema_full.py

judgelight:
	bash scripts/setup_judgelight.sh

figs:
	uv run python experiments/q2_decomposition/figures.py
	uv run python experiments/comparison/plot_ablation.py
	uv run python experiments/q4_end_to_end/plot_cactus.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage .hypothesis
