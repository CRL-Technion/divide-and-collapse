#!/usr/bin/env bash
# Stand up the isolated .venv-judgelight environment for running Tang et al.'s
# benchmark_greedy_vs_ilp.py and opt_main.py. Kept separate from the main .venv
# because pogema pins numpy<=1.26.4 and pydantic<=1.9.1, which conflict with
# the mapfc Python toolchain.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_ROOT/.venv-judgelight"

if [ ! -d "$VENV" ]; then
  python -m uv venv "$VENV" --python 3.12
fi

python -m uv pip install --python "$VENV/Scripts/python.exe" \
  "numpy>=1.23.5,<=1.26.4" \
  "pydantic>=1.8.2,<=1.9.1" \
  "pettingzoo==1.22.3" \
  "tabulate>=0.8.7,<=0.8.10" \
  "gymnasium==0.28.1" \
  "matplotlib<=3.8.3" \
  "PyYAML<=6.0.1" \
  "loguru<=0.7.2" \
  "pandas<=2.2.1" \
  "scipy" \
  "gurobipy==13.0.2"
