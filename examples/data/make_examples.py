"""Generate the small first-party example instances shipped in this folder.

These instances are written from scratch here so the repository ships no
third-party benchmark data; regenerate them at any time with::

    python examples/data/make_examples.py

Each output is a JSONL file in the format ``examples/solve_instance.py`` and
the experiment harness consume: one row per agent, with a grid-coordinate
trajectory. All three are small enough to check by hand, and each isolates a
different behaviour of the framework.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write(name: str, scenario: str, trajectories: list[list[tuple[int, int]]]) -> None:
    path = HERE / f"{name}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for idx, traj in enumerate(trajectories):
            fh.write(
                json.dumps(
                    {
                        "dataset": "examples",
                        "scenario": scenario,
                        "num_agents": len(trajectories),
                        "agent_idx": idx,
                        "trajectory": [list(xy) for xy in traj],
                    }
                )
                + "\n"
            )
    print(f"wrote {path.name} ({len(trajectories)} agents)")


def main() -> None:
    # 1. Singleton dispatch only. Two agents in separate corridors, each
    #    stepping out and back before moving on. Their reach sets are disjoint,
    #    so the interaction graph has no edges, both are singletons, and each is
    #    solved in linear time by the per-agent primitive. Cost 8 -> 4.
    write(
        "toy_singletons",
        "corridor",
        [
            [(0, 0), (0, 1), (0, 0), (0, 1), (0, 2)],
            [(2, 0), (2, 1), (2, 0), (2, 1), (2, 2)],
        ],
    )

    # 2. The paper's running example (Fig. 1) in grid coordinates, with the hub
    #    b = (1,1), a = (0,1), c = (2,1), e = (1,2), f = (1,0). Both agents want
    #    to park on the hub, so they form one non-trivial component and the
    #    joint solver must arbitrate. Cost 8 -> 4, and neither agent gets its
    #    unconstrained optimum.
    write(
        "toy_contention",
        "hub",
        [
            [(0, 1), (1, 1), (2, 1), (1, 1), (1, 2)],
            [(1, 1), (1, 0), (1, 1), (1, 0), (1, 1)],
        ],
    )

    # 3. A six-agent mix: one contending pair (as above) plus four agents on
    #    their own, so a single instance exercises singleton dispatch and the
    #    joint solver together.
    write(
        "toy_mixed",
        "mixed",
        [
            [(0, 1), (1, 1), (2, 1), (1, 1), (1, 2)],
            [(1, 1), (1, 0), (1, 1), (1, 0), (1, 1)],
            [(5, 5), (5, 6), (5, 5), (5, 6), (5, 7)],
            [(7, 5), (7, 6), (7, 5), (7, 6), (7, 7)],
            [(9, 0), (9, 1), (9, 0), (9, 1), (9, 2)],
            [(0, 9), (1, 9), (0, 9), (1, 9), (2, 9)],
        ],
    )


if __name__ == "__main__":
    main()
