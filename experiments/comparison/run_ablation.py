"""CCBS engineering ablation: incremental variants on the non-trivial residue.

Runs ``solve_ccbs`` in five configurations on every non-trivial component of
size <= ``--max-size`` from ``benchmarks/schedules/pogema_500.jsonl`` under a
common per-component budget, logging per-component runtime, solved flag, cost,
and high-level tree size (``pop_count``) for each variant. Unlike
``run_joint_comparison.py`` this is CCBS-internal only (no Judgelight), so it
runs entirely in the main venv with no subprocess batching.

Variants form an incremental ladder, each adding one App.~A ingredient:
  vanilla   : no engineering (earliest-time selection, classical CBS branching)
  cardinal  : + cardinal-conflict prioritisation (the production adaptive
              selector, cardinal_trigger_pops=0)
  disjoint  : + disjoint splitting (App.~A.4)
  full      : + conflict bypass (App.~A.5)  == production default

Writes ``results/raw/ablation_lazylb.parquet`` (one row per component x variant)
and prints per-variant solve rates and a variant x size-bucket median-runtime
table.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import polars as pl

from mapfc.decompose import decompose
from mapfc.joint.base import baseline_cost
from mapfc.joint.ccbs import solve_ccbs

REPO = Path(__file__).resolve().parents[2]
JSONL_IN = REPO / "benchmarks" / "schedules" / "pogema_full.jsonl"
PARQUET_OUT = REPO / "results" / "raw" / "ablation_lazylb_full.parquet"

DEFAULT_MAX_SIZE = 50
DEFAULT_TIME_LIMIT = 5.0

# (name, solve_ccbs kwargs). cardinal_trigger_pops: a very large value = A.2
# never activates (earliest-time selection); 16 = the production adaptive
# selector (defers to cardinal-first once the high-level pop count exceeds 16).
VARIANTS: list[tuple[str, dict[str, object]]] = [
    ("vanilla", dict(cardinal_trigger_pops=10**9, use_disjoint_split=False, use_bypass=False)),
    ("cardinal", dict(cardinal_trigger_pops=0, use_disjoint_split=False, use_bypass=False)),
    ("disjoint", dict(cardinal_trigger_pops=0, use_disjoint_split=True, use_bypass=False)),
    ("full", dict(cardinal_trigger_pops=0, use_disjoint_split=True, use_bypass=True)),
]


def _load_components(max_size: int):
    by_instance: dict[tuple[str, str, int], dict[int, tuple]] = defaultdict(dict)
    with JSONL_IN.open(encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            key = (entry["dataset"], entry["scenario"], entry["num_agents"])
            by_instance[key][entry["agent_idx"]] = tuple(tuple(xy) for xy in entry["trajectory"])
    records = []
    skipped = 0
    for (dataset, scenario, num_agents), traj_by_idx in sorted(by_instance.items()):
        schedules = [traj_by_idx[i] for i in sorted(traj_by_idx)]
        decomp = decompose(schedules)
        for comp in decomp.non_trivial_components:
            if len(comp) > max_size:
                skipped += 1
                continue
            sorted_ids = tuple(sorted(comp))
            sub = tuple(schedules[i] for i in sorted_ids)
            records.append((dataset, scenario, num_agents, sorted_ids, sub))
    if skipped:
        print(f"  skipped {skipped} components with size > {max_size}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE)
    parser.add_argument("--time-limit-sec", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--out", type=Path, default=PARQUET_OUT)
    args = parser.parse_args()

    print(f"loading components (max_size={args.max_size})...", flush=True)
    records = _load_components(args.max_size)
    print(f"  {len(records)} non-trivial components in scope", flush=True)

    rows: list[dict[str, object]] = []
    for vname, kwargs in VARIANTS:
        print(f"running variant '{vname}' {kwargs}...", flush=True)
        t0 = time.perf_counter()
        n_timeout = 0
        for dataset, scenario, num_agents, ids, schedules in records:
            diag: dict[str, object] = {}
            tic = time.perf_counter()
            result = solve_ccbs(
                list(schedules),
                time_limit_sec=args.time_limit_sec,
                diagnostics=diag,
                **kwargs,
            )
            elapsed = time.perf_counter() - tic
            solved = result is not None
            if not solved:
                n_timeout += 1
            rows.append(
                {
                    "variant": vname,
                    "dataset": dataset,
                    "scenario": scenario,
                    "num_agents": num_agents,
                    "comp_size": len(ids),
                    "baseline_cost": int(baseline_cost(schedules)),
                    "runtime_s": elapsed,
                    "solved": solved,
                    "cost": (None if result is None else int(result.cost)),
                    "pop_count": int(diag.get("pop_count", 0)),
                }
            )
        print(
            f"  '{vname}' done in {time.perf_counter() - t0:.1f}s ({n_timeout} timeouts)",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(args.out)
    print(f"wrote {args.out} ({len(rows)} rows)", flush=True)

    print("\n## Solve rate by variant")
    for vname, _ in VARIANTS:
        sub = [r for r in rows if r["variant"] == vname]
        ok = sum(1 for r in sub if r["solved"])
        print(f"  {vname:>9}: {ok}/{len(sub)} ({100 * ok / len(sub):.1f}%)")

    print("\n## Median runtime (ms) by variant x size bucket")
    buckets = [(2, 4), (5, 9), (10, 19), (20, 29), (30, 39), (40, 50)]
    print("  " + "variant".rjust(9) + "".join(f"{f'{lo}-{hi}':>10}" for lo, hi in buckets))
    for vname, _ in VARIANTS:
        cells = []
        for lo, hi in buckets:
            sub = [r for r in rows if r["variant"] == vname and lo <= int(r["comp_size"]) <= hi]
            if sub:
                med = statistics.median(float(r["runtime_s"]) for r in sub) * 1e3
                cells.append(f"{med:>9.2f} ")
            else:
                cells.append(f"{'-':>9} ")
        print("  " + vname.rjust(9) + "".join(cells))


if __name__ == "__main__":
    main()
