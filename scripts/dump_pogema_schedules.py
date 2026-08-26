"""Dump POGEMA-derived single-agent schedules for the Q1 benchmark.

Runs in ``.venv-judgelight`` (depends on the pinned pogema + pogema-benchmark
+ pogema-toolbox stack). For each ``(dataset, scenario, num_agents)`` instance
selected by the caller, the script:

1. Builds a POGEMA ``GridConfig`` matching Tang et al.'s
   ``benchmark_greedy_vs_ilp.build_env_config``.
2. Runs ``BatchAStarAgent`` against ``TrajectoryCollectorWrapper`` until every
   agent terminates or the episode is truncated.
3. Records the per-agent trajectory ``M^i`` (a list of ``(row, col)`` tuples).
4. Emits one JSON object per agent to a JSONL file at ``--out``.

The output schema is:

    {
        "dataset": "02-mazes",
        "scenario": "Scenario-640",
        "num_agents": 8,
        "agent_idx": 0,
        "k_i": 22,                          # |M^i| = len(trajectory) - 1
        "trajectory": [[r, c], [r, c], ...] # M^i as a list of (row, col)
    }

By default the script runs ``--per-cell`` random scenarios per
``(dataset, num_agents)`` cell, sampled with a fixed seed for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JUDGELIGHT = REPO_ROOT / "third_party" / "judgelight"
sys.path.insert(0, str(JUDGELIGHT / "pogema"))
sys.path.insert(0, str(JUDGELIGHT / "pogema-toolbox"))
sys.path.insert(0, str(JUDGELIGHT / "pogema-benchmark" / "algorithms"))

from pogema import BatchAStarAgent, GridConfig, pogema_v0  # noqa: E402
from trajectory_collector import TrajectoryCollectorWrapper  # noqa: E402

from mapfc.io.judgelight_dataset import DATASETS, Instance, load_dataset  # noqa: E402

DATASET_ROOT = JUDGELIGHT / "pogema-benchmark" / "algorithms" / "experiments"
DEFAULT_SEED = 20260621


def _build_env_config(instance: Instance) -> GridConfig:
    map_str = "\n".join(instance.grid)
    return GridConfig(
        on_target="nothing",
        collision_system="soft",
        observation_type="MAPF",
        max_episode_steps=128,
        map=map_str,
        map_name=instance.map_name,
        seed=instance.seed,
        num_agents=instance.num_agents,
        agents_xy=[list(xy) for xy in instance.agents_xy],
        targets_xy=[list(xy) for xy in instance.targets_xy],
    )


def _run_episode(instance: Instance) -> list[list[tuple[int, int]]]:
    env = pogema_v0(grid_config=_build_env_config(instance))
    env = TrajectoryCollectorWrapper(env)
    algo = BatchAStarAgent()
    algo.reset_states()
    obs, _ = env.reset(seed=instance.seed)
    while True:
        obs, _rew, terminated, truncated, _infos = env.step(algo.act(obs))
        if all(terminated) or all(truncated):
            break
    return env.get_trajectories()


def _sample_instances(dataset: str, per_cell: int, rng: random.Random) -> list[Instance]:
    by_cell: dict[int, list[Instance]] = {}
    for inst in load_dataset(DATASET_ROOT / dataset):
        by_cell.setdefault(inst.num_agents, []).append(inst)
    sampled: list[Instance] = []
    for k in sorted(by_cell):
        pool = by_cell[k]
        rng.shuffle(pool)
        sampled.extend(pool[:per_cell])
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(DATASETS),
        help="Datasets to sample from. Default: all five MAPF families.",
    )
    parser.add_argument(
        "--per-cell",
        type=int,
        default=2,
        help="Scenarios to sample per (dataset, num_agents) cell.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "schedules" / "pogema_subset.jsonl",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    t_overall = time.time()
    written = 0
    failed: list[tuple[str, str, int, str]] = []

    with args.out.open("w", encoding="utf-8") as fh:
        for dataset in args.datasets:
            sampled = _sample_instances(dataset, args.per_cell, rng)
            t_ds = time.time()
            for inst in sampled:
                try:
                    trajectories = _run_episode(inst)
                except Exception as exc:
                    failed.append((dataset, inst.scenario_name, inst.num_agents, repr(exc)))
                    continue
                for i, traj in enumerate(trajectories):
                    record = {
                        "dataset": dataset,
                        "scenario": inst.scenario_name,
                        "num_agents": inst.num_agents,
                        "agent_idx": i,
                        "k_i": max(len(traj) - 1, 0),
                        "trajectory": [list(xy) for xy in traj],
                    }
                    fh.write(json.dumps(record) + "\n")
                    written += 1
            print(f"{dataset}: {len(sampled)} scenarios in {time.time() - t_ds:.1f}s")

    print(f"\nwrote {written} agent-trajectories to {args.out}")
    print(f"total wall-clock: {time.time() - t_overall:.1f}s")
    if failed:
        print(f"FAILED: {len(failed)} instances")
        for entry in failed[:10]:
            print(f"  {entry}")


if __name__ == "__main__":
    main()
