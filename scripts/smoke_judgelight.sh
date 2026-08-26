#!/usr/bin/env bash
# Reproduce Tang et al.'s smoke test on 02-mazes/Scenario-640 with num_agents=8
# and diff the SoC metrics against the shipped benchmark_outputs_test/ dump.
# Hardware-dependent timing fields are compared as a ratio, not an equality.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JUDGELIGHT="$REPO_ROOT/third_party/judgelight"
VENV_PY="$REPO_ROOT/.venv-judgelight/Scripts/python.exe"
OUT_REL="benchmark_outputs_mapfc_smoke"

if [ ! -x "$VENV_PY" ]; then
  echo "$VENV_PY missing — run scripts/setup_judgelight_venv.sh first" >&2
  exit 1
fi

cd "$JUDGELIGHT"
PYTHONUTF8=1 "$VENV_PY" benchmark_greedy_vs_ilp.py \
  --dataset 02-mazes \
  --num-scenarios 1 \
  --num-agents 8 \
  --out-dir "$OUT_REL"

"$VENV_PY" - <<PY
import json
from pathlib import Path
ours = json.loads(Path("$OUT_REL/greedy_vs_ilp_results.json").read_text())
ref  = json.loads(Path("benchmark_outputs_test/greedy_vs_ilp_results.json").read_text())
DETERMINISTIC = {"dataset","scenario","map_name","num_agents",
                 "original_move_soc","ilp_move_soc","greedy_move_soc",
                 "ilp_saving","greedy_saving","saving_gap",
                 "ilp_actions","greedy_actions","num_ilp_actions","num_ilp_constraints",
                 "ilp_status","ilp_valid","greedy_valid"}
ok = True
for k in DETERMINISTIC:
    if ours[0][k] != ref[0][k]:
        print(f"DIFF {k}: ours={ours[0][k]} ref={ref[0][k]}")
        ok = False
print("Deterministic fields match" if ok else "MISMATCH — version drift")
print(f"  Tang ilp_time  = {ref[0]['ilp_time']:.4f}s")
print(f"  Our  ilp_time  = {ours[0]['ilp_time']:.4f}s")
print(f"  Ratio (ours/Tang) = {ours[0]['ilp_time']/ref[0]['ilp_time']:.3f}")
PY
