"""Q2 — singleton-majority hypothesis test on the POGEMA subset.

Consumes ``benchmarks/schedules/pogema_subset.jsonl`` (one record per agent),
groups records into instances by ``(dataset, scenario, num_agents)``, runs
:func:`mapfc.decompose.decompose` on each instance, and emits

- ``results/raw/q2_decomposition.parquet`` — one row per instance with
  singleton count, non-trivial component sizes, edges_count, reach total,
  H-construction wall-clock.
- A markdown summary on stdout: per-family singleton fraction (mean and
  median over instances), max non-trivial component size, mean H build
  time, and instance count.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import polars as pl

from mapfc.decompose import decompose

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO_ROOT / "benchmarks" / "schedules" / "pogema_subset.jsonl"
DEFAULT_OUT = REPO_ROOT / "results" / "raw" / "q2_decomposition.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    jsonl_in = args.in_path
    parquet_out = args.out_path
    if not jsonl_in.exists():
        raise FileNotFoundError(f"{jsonl_in} not found; run scripts/dump_pogema_schedules.py first")

    by_instance: dict[tuple[str, str, int], dict[int, list[tuple[int, int]]]] = defaultdict(dict)
    with jsonl_in.open(encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            key = (entry["dataset"], entry["scenario"], entry["num_agents"])
            agent_idx = entry["agent_idx"]
            traj = [tuple(xy) for xy in entry["trajectory"]]
            by_instance[key][agent_idx] = traj

    rows: list[dict[str, object]] = []
    for (dataset, scenario, num_agents), traj_by_idx in sorted(by_instance.items()):
        schedules = [traj_by_idx[i] for i in sorted(traj_by_idx)]
        result = decompose(schedules)
        non_trivial_sizes = sorted((len(c) for c in result.non_trivial_components), reverse=True)
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "num_agents": num_agents,
                "n_singletons": result.n_singletons,
                "n_non_trivial": len(result.non_trivial_components),
                "singleton_fraction": result.singleton_fraction,
                "max_non_trivial_size": result.max_non_trivial_size,
                "non_trivial_sizes": str(non_trivial_sizes),
                "edges_count": result.edges_count,
                "reach_total_cells": result.reach_total_cells,
                "h_construction_time_s": result.h_construction_time_s,
            }
        )

    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows)
    df.write_parquet(parquet_out)
    print(f"wrote {parquet_out} ({len(rows)} instances)")

    print("\n## Q2 — singleton-majority hypothesis per family")
    print(
        f"{'family':>14} {'#inst':>6} {'agents':>8} {'singleton%':>12} "
        f"{'median %':>10} {'max NT size':>14} {'H_us_med':>10}"
    )
    for fam in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == fam]
        n_inst = len(sub)
        total_agents = sum(int(r["num_agents"]) for r in sub)
        total_singletons = sum(int(r["n_singletons"]) for r in sub)
        agg_pct = 100.0 * total_singletons / total_agents if total_agents else 0.0
        med_pct = 100.0 * statistics.median(float(r["singleton_fraction"]) for r in sub)
        max_nt = max(int(r["max_non_trivial_size"]) for r in sub)
        h_med_us = statistics.median(float(r["h_construction_time_s"]) for r in sub) * 1e6
        print(
            f"{fam:>14} {n_inst:>6} {total_agents:>8} {agg_pct:>11.2f}% "
            f"{med_pct:>9.2f}% {max_nt:>14} {h_med_us:>10.1f}"
        )

    print("\n## Component-size histogram across the entire subset")
    histogram: dict[int, int] = {}
    for r in rows:
        histogram[1] = histogram.get(1, 0) + int(r["n_singletons"])
        sizes = eval(str(r["non_trivial_sizes"]))
        for s in sizes:
            histogram[int(s)] = histogram.get(int(s), 0) + 1
    total_components = sum(histogram.values())
    print(f"{'size':>6} {'count':>8} {'fraction':>10}")
    for size in sorted(histogram):
        count = histogram[size]
        print(f"{size:>6} {count:>8} {count / total_components:>10.4f}")


if __name__ == "__main__":
    main()
