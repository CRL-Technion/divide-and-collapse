"""Solve a MAPF-Collapse instance from a JSONL file and report the saving.

The input format is the one the experiment harness consumes: one JSON object
per agent, with the fields

    {"dataset": str, "scenario": str, "num_agents": int,
     "agent_idx": int, "trajectory": [[x, y], [x, y], ...]}

Agents sharing a ``(dataset, scenario, num_agents)`` triple form one instance.
``scripts/dump_pogema_full.py`` writes this format for the POGEMA suite; the
files under ``examples/data/`` are small hand-built instances that ship with
the repository.

Examples::

    python examples/solve_instance.py examples/data/toy_corridor.jsonl
    python examples/solve_instance.py examples/data/toy_corridor.jsonl --config mapfc+ccbs+jl+gate40
    python examples/solve_instance.py mydata.jsonl --config nocollapse

Pass ``--list-configs`` to see every available configuration. Configurations
whose name contains ``judgelight`` additionally require Tang et al.'s solver;
see THIRD-PARTY.md and ``scripts/setup_judgelight.sh``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from mapfc.eval import ALL_CONFIG_NAMES, CONFIG_BY_NAME, run_config


def load_instances(path: Path):
    """Group the JSONL rows into ``(dataset, scenario, num_agents) -> schedules``."""
    by_instance: dict[tuple[str, str, int], dict[int, tuple]] = defaultdict(dict)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            key = (entry["dataset"], entry["scenario"], entry["num_agents"])
            by_instance[key][entry["agent_idx"]] = tuple(tuple(xy) for xy in entry["trajectory"])
    for key, traj_by_idx in sorted(by_instance.items()):
        yield key, tuple(traj_by_idx[i] for i in sorted(traj_by_idx))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, nargs="?", help="input instances")
    parser.add_argument(
        "--config",
        default="mapfc+ccbs",
        help="configuration to run (default: mapfc+ccbs, the exact solver)",
    )
    parser.add_argument("--budget-sec", type=float, default=30.0, help="per-instance time limit")
    parser.add_argument("--list-configs", action="store_true")
    args = parser.parse_args()

    if args.list_configs:
        print("Available configurations:")
        for name in ALL_CONFIG_NAMES:
            note = "  (needs Judgelight)" if "judgelight" in name or name.endswith("jl") else ""
            print(f"  {name}{note}")
        return

    if args.jsonl is None:
        parser.error("give an input .jsonl (or --list-configs)")
    if args.config not in CONFIG_BY_NAME:
        parser.error(f"unknown config {args.config!r}; try --list-configs")

    cfg = CONFIG_BY_NAME[args.config]
    print(f"config: {args.config}   budget: {args.budget_sec}s\n")
    print(
        f"{'instance':<28} {'agents':>7} {'singl.':>7} {'comps':>6} "
        f"{'cost':>6} {'saving':>7} {'time':>11}"
    )

    for (dataset, scenario, num_agents), schedules in load_instances(args.jsonl):
        rec, _ = run_config(
            cfg,
            dataset=dataset,
            scenario=scenario,
            num_agents=num_agents,
            schedules=schedules,
            budget_s=args.budget_sec,
        )
        label = f"{dataset}/{scenario}/n{num_agents}"
        if rec.soc is None:
            print(
                f"{label:<28} {num_agents:>7} {rec.singleton_count:>7} "
                f"{rec.nontrivial_count:>6} {'--':>6} {'--':>7} {'timeout':>11}"
            )
            continue
        print(
            f"{label:<28} {num_agents:>7} {rec.singleton_count:>7} "
            f"{rec.nontrivial_count:>6} {rec.soc:>6} "
            f"{rec.saving_ratio:>7.3f} {rec.runtime_total_s * 1e3:>8.1f} ms"
        )


if __name__ == "__main__":
    main()
