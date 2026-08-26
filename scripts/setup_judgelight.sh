#!/usr/bin/env bash
# Recursively clone Tang et al.'s Judgelight and pin to the SHAs frozen in
# third_party/SUBMODULE_SHAS.json. Re-run after a SHAs bump.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$REPO_ROOT/third_party/judgelight"
SHAS="$REPO_ROOT/third_party/SUBMODULE_SHAS.json"

if ! command -v python >/dev/null 2>&1; then
  echo "python not on PATH" >&2; exit 1
fi

if [ -d "$TARGET/.git" ]; then
  echo "[setup_judgelight] $TARGET already exists — pulling pinned SHAs"
else
  echo "[setup_judgelight] recursive clone into $TARGET"
  git clone --recurse-submodules https://github.com/TachikakaMin/Judgelight.git "$TARGET"
fi

python - "$SHAS" "$TARGET" <<'PY'
import json, subprocess, sys
shas_path, target = sys.argv[1], sys.argv[2]
shas = json.load(open(shas_path, encoding="utf-8"))

def run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True)

run(["git", "fetch", "--all", "--tags"], cwd=target)
run(["git", "checkout", shas["judgelight"]["sha"]], cwd=target)

for name, info in shas["nested_submodules"].items():
    sub = f"{target}/{name}"
    run(["git", "fetch", "--all", "--tags"], cwd=sub)
    run(["git", "checkout", info["sha"]], cwd=sub)
print("[setup_judgelight] pinned SHAs applied")
PY
