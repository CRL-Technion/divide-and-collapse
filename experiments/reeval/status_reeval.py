"""Poll the re-evaluation harness progress without touching the running job.

Reads the ``reeval_<scope>_progress.json`` written by ``run_reeval.py`` after
every instance and prints a one-shot human-readable status: overall percent,
completed/total cells, per-config progress, elapsed, throughput, and ETA.

Usage::

    .venv/Scripts/python.exe experiments/reeval/status_reeval.py            # scope 500
    .venv/Scripts/python.exe experiments/reeval/status_reeval.py --scope full
    .venv/Scripts/python.exe experiments/reeval/status_reeval.py --watch    # refresh every 10 s
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "results" / "raw"


def _fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _print_status(progress_path: Path) -> bool:
    if not progress_path.exists():
        print(f"no progress file yet at {progress_path}")
        return False
    try:
        p = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("progress file is mid-write; try again in a moment")
        return False

    staleness = time.time() - p.get("updated_at", 0)
    print(
        f"[{p.get('status', '?')}] scope={p.get('scope')}  "
        f"{p.get('percent', 0):.1f}%  "
        f"{p.get('completed_cells', 0)}/{p.get('total_cells', 0)} cells"
    )
    print(
        f"  elapsed {_fmt_dur(p.get('elapsed_s'))}  "
        f"eta {_fmt_dur(p.get('eta_s'))}  "
        f"rate {p.get('rate_cells_per_s', 0):.2f} cells/s  "
        f"current={p.get('current_config')}  "
        f"(updated {_fmt_dur(staleness)} ago)"
    )
    for name, v in p.get("per_config", {}).items():
        done, total = v.get("done", 0), v.get("total", 0)
        bar_n = int(20 * done / total) if total else 0
        bar = "#" * bar_n + "-" * (20 - bar_n)
        print(f"    {name:<22} [{bar}] {done}/{total}")
    return p.get("status") == "done"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="500")
    parser.add_argument("--progress", type=Path, default=None)
    parser.add_argument("--watch", action="store_true", help="Refresh every --interval seconds.")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()

    progress_path = args.progress or (RESULTS_DIR / f"reeval_{args.scope}_progress.json")
    if not args.watch:
        _print_status(progress_path)
        return
    try:
        while True:
            print("\033[2J\033[H", end="")  # clear screen
            done = _print_status(progress_path)
            if done:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
