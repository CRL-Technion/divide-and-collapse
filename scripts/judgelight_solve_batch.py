"""Batched subprocess driver for Tang et al.'s Judgelight ILP solver.

Same contract as ``scripts/judgelight_solve.py`` but reads a list of items
(each item carries an ``id`` and a single-agent or joint ``M``) and emits a
list of results. Amortises subprocess + import overhead across many calls.

Input schema:

    {
        "items": [
            {"id": <int>, "M": [...]},
            ...
        ],
        "time_limit_sec": 5.0,
        "threads": 1
    }

Output schema:

    {
        "results": [
            {
                "id": <int>,
                "M_opt": [...],
                "best_min_cost": <int>,
                "ilp_saving": <int>,
                "preprocess_saving": <int>,
                "total_runtime_s": <float>
            },
            ...
        ]
    }
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


def _to_hashable(v: object) -> object:
    return tuple(v) if isinstance(v, list) else v


@contextlib.contextmanager
def _redirect_stdout_fd_to_stderr() -> Iterator[None]:
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def main() -> None:
    payload = json.load(sys.stdin)
    items = payload["items"]
    time_limit = float(payload.get("time_limit_sec", 5.0))
    threads = int(payload.get("threads", 1))

    results: list[dict[str, object]] = []
    with _redirect_stdout_fd_to_stderr():
        for item in items:
            iid = item["id"]
            M_in = [[_to_hashable(v) for v in agent] for agent in item["M"]]
            t0 = time.perf_counter()
            summary, chosen = solve_collapsed_mapf_from_M(
                M_in,
                time_limit_sec=time_limit,
                threads=threads,
                verbose=False,
            )
            M_opt = apply_actions_to_trajectory(summary["M_preprocessed"], chosen)
            elapsed = time.perf_counter() - t0
            M_opt_jsonable = [
                [list(v) if isinstance(v, tuple) else v for v in agent] for agent in M_opt
            ]
            results.append(
                {
                    "id": iid,
                    "M_opt": M_opt_jsonable,
                    "best_min_cost": int(summary.get("best_min_cost", 0)),
                    "ilp_saving": int(summary.get("ilp_saving", 0)),
                    "preprocess_saving": int(summary.get("preprocess_saving", 0)),
                    "total_runtime_s": float(elapsed),
                }
            )

    json.dump({"results": results}, sys.stdout)


if __name__ == "__main__":
    main()
