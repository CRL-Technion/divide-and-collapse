"""Subprocess worker for Tang et al.'s Judgelight ILP solver.

Runs a request-response loop: one JSON request per line from stdin, one
JSON response per line to stdout. This amortises Python interpreter
startup + Gurobi library load + license check across many MAPF-Collapse
sub-instances — see DECISIONS.md "subprocess-tax diagnostic" entry,
where the fixed overhead measured to ~500 ms per fresh invocation, and
the original wrapper paid that tax once per non-trivial component
(2-4x per instance on POGEMA, dominating the wall-clock budget).

Request line:

    {"op": "solve", "M": [...], "time_limit_sec": 30.0, "threads": 2}
    {"op": "shutdown"}

Response line:

    {"ok": true,  "M_opt": [...], "best_min_cost": ..., ...}
    {"ok": false, "error": "..."}

EOF on stdin terminates the worker gracefully. The legacy single-shot
format (one bare JSON document with "M" at the top level, no trailing
newline) is also accepted for backward compatibility with any caller
that still uses ``subprocess.run([...], input=json.dumps(...))``.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JUDGELIGHT = REPO_ROOT / "third_party" / "judgelight"
sys.path.insert(0, str(JUDGELIGHT))
sys.path.insert(0, str(JUDGELIGHT / "pogema"))
sys.path.insert(0, str(JUDGELIGHT / "pogema-toolbox"))
sys.path.insert(0, str(JUDGELIGHT / "pogema-benchmark" / "algorithms"))

from opt_main import (  # noqa: E402
    apply_actions_to_trajectory,
    solve_collapsed_mapf_from_M,
)


def _to_hashable(vertex: object) -> object:
    return tuple(vertex) if isinstance(vertex, list) else vertex


@contextlib.contextmanager
def _redirect_stdout_fd_to_stderr() -> Iterator[None]:
    """Redirect the OS-level stdout file descriptor to stderr.

    Gurobi's C runtime writes its WLS-license header and other status
    messages to file-descriptor 1, bypassing Python's ``sys.stdout``.
    Redirecting at the FD level keeps the response stream clean.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def _solve_one(payload: dict) -> dict:
    """Run one solve, return a response dict."""
    M_in = [[_to_hashable(v) for v in agent] for agent in payload["M"]]
    time_limit = float(payload.get("time_limit_sec", 30.0))
    threads = int(payload.get("threads", 2))

    t_total_start = time.perf_counter()
    with _redirect_stdout_fd_to_stderr():
        summary, chosen = solve_collapsed_mapf_from_M(
            M_in,
            time_limit_sec=time_limit,
            threads=threads,
            verbose=False,
        )
        M_opt = apply_actions_to_trajectory(summary["M_preprocessed"], chosen)
    total_s = time.perf_counter() - t_total_start

    M_opt_jsonable = [
        [list(v) if isinstance(v, tuple) else v for v in agent] for agent in M_opt
    ]
    solve_runtime = float(summary.get("solve_time_sec", 0.0)) or float(
        summary.get("optimization_time", 0.0)
    )
    return {
        "ok": True,
        "M_opt": M_opt_jsonable,
        "best_min_cost": int(summary.get("best_min_cost", 0)),
        "ilp_saving": int(summary.get("ilp_saving", 0)),
        "preprocess_saving": int(summary.get("preprocess_saving", 0)),
        "build_runtime_s": max(0.0, total_s - solve_runtime),
        "solve_runtime_s": solve_runtime,
        "total_runtime_s": total_s,
    }


def _emit(response: dict) -> None:
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def main() -> None:
    while True:
        line = sys.stdin.readline()
        if not line:
            return  # EOF -> graceful shutdown
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            _emit({"ok": False, "error": f"bad JSON: {exc}"})
            continue

        op = payload.get("op", "solve")  # legacy callers omit "op"
        if op == "shutdown":
            return
        if op != "solve":
            _emit({"ok": False, "error": f"unknown op: {op!r}"})
            continue

        try:
            response = _solve_one(payload)
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _emit(response)


if __name__ == "__main__":
    main()
