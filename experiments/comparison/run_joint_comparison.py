"""CCBS vs Judgelight on every non-trivial component in the POGEMA subset.

Decomposes each instance in ``benchmarks/schedules/pogema_500.jsonl``,
extracts every non-trivial component, and runs both solvers under a
common per-component time budget (default 5 s, matching the implementation
plan). CCBS runs natively in the main venv; Judgelight is batched into
``scripts/judgelight_solve_batch.py`` so the ~300 ms subprocess startup
amortises across all components.

Writes ``results/raw/joint_comparison.parquet`` with one row per component
and a stdout summary covering cost agreement and per-size-bucket success
rates.

Large components (size > ``--max-size``, default 50) are skipped to keep
the total wall-clock bounded; the cap is reported with the result.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import polars as pl

from mapfc.decompose import decompose
from mapfc.joint.base import baseline_cost
from mapfc.joint.ccbs import solve_ccbs

REPO = Path(__file__).resolve().parents[2]
JSONL_IN = REPO / "benchmarks" / "schedules" / "pogema_full.jsonl"
PARQUET_OUT = REPO / "results" / "raw" / "joint_comparison_full.parquet"
VENV_PY = REPO / ".venv-judgelight" / "Scripts" / "python.exe"
BATCH_SCRIPT = REPO / "scripts" / "judgelight_solve_batch.py"

JUDGELIGHT_BATCH = 50
DEFAULT_MAX_SIZE = 50
DEFAULT_TIME_LIMIT = 5.0


def _load_components(
    min_size: int,
    max_size: int,
) -> list[tuple[str, str, int, tuple[int, ...], tuple[tuple[tuple[int, int], ...], ...]]]:
    """Return (dataset, scenario, num_agents, component_member_ids, schedules)
    for every non-trivial component of every instance with
    min_size <= size <= max_size.
    """
    by_instance: dict[tuple[str, str, int], dict[int, tuple[tuple[int, int], ...]]] = defaultdict(
        dict
    )
    with JSONL_IN.open(encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            key = (entry["dataset"], entry["scenario"], entry["num_agents"])
            by_instance[key][entry["agent_idx"]] = tuple(tuple(xy) for xy in entry["trajectory"])

    records: list[
        tuple[
            str,
            str,
            int,
            tuple[int, ...],
            tuple[tuple[tuple[int, int], ...], ...],
        ]
    ] = []
    skipped = 0
    for (dataset, scenario, num_agents), traj_by_idx in sorted(by_instance.items()):
        schedules = [traj_by_idx[i] for i in sorted(traj_by_idx)]
        decomp = decompose(schedules)
        for comp in decomp.non_trivial_components:
            if not (min_size <= len(comp) <= max_size):
                skipped += 1
                continue
            sorted_ids = tuple(sorted(comp))
            sub_schedules = tuple(schedules[i] for i in sorted_ids)
            records.append((dataset, scenario, num_agents, sorted_ids, sub_schedules))
    if skipped:
        print(f"  skipped {skipped} components with size outside [{min_size}, {max_size}]")
    return records


def _run_ccbs_one(
    schedules: tuple[tuple[tuple[int, int], ...], ...],
    time_limit_sec: float,
) -> tuple[int | None, float]:
    """Wraps solve_ccbs to capture cost + wall-clock even on failure."""
    t0 = time.perf_counter()
    result = solve_ccbs(list(schedules), time_limit_sec=time_limit_sec)
    elapsed = time.perf_counter() - t0
    if result is None:
        return None, elapsed
    return result.cost, elapsed


def _run_judgelight_batch(
    records: list[
        tuple[
            str,
            str,
            int,
            tuple[int, ...],
            tuple[tuple[tuple[int, int], ...], ...],
        ]
    ],
    time_limit_sec: float,
) -> list[tuple[int | None, float, int | None]]:
    """Returns one (cost, total_runtime_s, ilp_status) per record."""
    out: list[tuple[int | None, float, int | None]] = [(None, 0.0, None)] * len(records)
    chunks = [
        list(range(i, min(i + JUDGELIGHT_BATCH, len(records))))
        for i in range(0, len(records), JUDGELIGHT_BATCH)
    ]
    print(f"  judgelight: {len(records)} components in {len(chunks)} chunks")

    env = {**os.environ, "PYTHONUTF8": "1"}
    for ci, chunk_ids in enumerate(chunks):
        items = []
        for idx in chunk_ids:
            schedules = records[idx][4]
            horizon = max(len(s) for s in schedules)
            padded = []
            for sched in schedules:
                last = sched[-1] if sched else None
                pad = list(sched) + [last] * (horizon - len(sched))
                padded.append([list(v) for v in pad])
            items.append({"id": idx, "M": padded})
        payload = {
            "items": items,
            "time_limit_sec": time_limit_sec,
            "threads": 1,
        }
        t0 = time.perf_counter()
        proc = subprocess.run(
            [str(VENV_PY), str(BATCH_SCRIPT)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            sys.stderr.write(f"chunk {ci} failed; stderr tail:\n{proc.stderr[-2000:]}\n")
            continue
        try:
            response = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            sys.stderr.write(f"chunk {ci} bad JSON: {exc}\n")
            continue
        for r in response["results"]:
            out[r["id"]] = (
                int(r["best_min_cost"]),
                float(r["total_runtime_s"]),
                None,
            )
        print(f"  chunk {ci + 1}/{len(chunks)}: {len(chunk_ids)} components " f"in {elapsed:.1f}s")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-size", type=int, default=2)
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE)
    parser.add_argument("--time-limit-sec", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--out", type=Path, default=PARQUET_OUT)
    args = parser.parse_args()

    print(f"loading components (size range [{args.min_size}, {args.max_size}])...")
    records = _load_components(args.min_size, args.max_size)
    print(f"  {len(records)} non-trivial components in scope")

    print(f"running CCBS (time_limit_sec={args.time_limit_sec})...")
    t0 = time.perf_counter()
    ccbs_results: list[tuple[int | None, float]] = []
    for i, (_ds, _sc, _na, _ids, schedules) in enumerate(records):
        if i % 100 == 0 and i > 0:
            print(f"  CCBS [{i}/{len(records)}]")
        ccbs_results.append(_run_ccbs_one(schedules, args.time_limit_sec))
    print(f"  CCBS done in {time.perf_counter() - t0:.1f}s")

    print(f"running Judgelight (time_limit_sec={args.time_limit_sec})...")
    t0 = time.perf_counter()
    jl_results = _run_judgelight_batch(records, args.time_limit_sec)
    print(f"  Judgelight done in {time.perf_counter() - t0:.1f}s")

    rows: list[dict[str, object]] = []
    for i, (dataset, scenario, num_agents, ids, schedules) in enumerate(records):
        ccbs_cost, ccbs_s = ccbs_results[i]
        jl_cost, jl_s, _ = jl_results[i]
        base = baseline_cost(schedules)  # NoCollapse SoC of this component
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "num_agents": num_agents,
                "comp_size": len(ids),
                "member_ids": list(ids),
                "baseline_cost": int(base),
                "ccbs_cost": ccbs_cost,
                "ccbs_s": ccbs_s,
                "ccbs_success": ccbs_cost is not None,
                "judgelight_cost": jl_cost,
                "judgelight_s": jl_s,
                "judgelight_success": jl_cost is not None,
                "both_succeeded": (ccbs_cost is not None and jl_cost is not None),
                "cost_match": (
                    ccbs_cost is not None and jl_cost is not None and ccbs_cost == jl_cost
                ),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows)
    df.write_parquet(args.out)
    print(f"wrote {args.out} ({len(rows)} rows)")

    print("\n## Aggregate")
    n = len(rows)
    ccbs_ok = sum(1 for r in rows if r["ccbs_success"])
    jl_ok = sum(1 for r in rows if r["judgelight_success"])
    both_ok = sum(1 for r in rows if r["both_succeeded"])
    matched = sum(1 for r in rows if r["cost_match"])
    print(f"CCBS success rate:        {ccbs_ok}/{n} ({100 * ccbs_ok / n:.1f}%)")
    print(f"Judgelight success rate:  {jl_ok}/{n} ({100 * jl_ok / n:.1f}%)")
    print(f"Both succeeded:           {both_ok}/{n}")
    if both_ok:
        print(f"Cost agreement (both):    {matched}/{both_ok} " f"({100 * matched / both_ok:.2f}%)")
    ccbs_lower = sum(
        1
        for r in rows
        if r["both_succeeded"]
        and r["ccbs_cost"] is not None
        and r["judgelight_cost"] is not None
        and int(r["ccbs_cost"]) < int(r["judgelight_cost"])
    )
    if both_ok:
        print(f"CCBS strictly better:     {ccbs_lower}/{both_ok}")

    print("\n## By component size")
    print(
        f"{'size':>8} {'#comp':>6} {'ccbs_ok':>8} {'jl_ok':>8} " f"{'ccbs_med':>10} {'jl_med':>10}"
    )
    buckets = [(2, 4), (5, 9), (10, 19), (20, 50)]
    for lo, hi in buckets:
        sub = [r for r in rows if lo <= int(r["comp_size"]) <= hi]
        if not sub:
            continue
        c_ok = sum(1 for r in sub if r["ccbs_success"])
        j_ok = sum(1 for r in sub if r["judgelight_success"])
        c_med = statistics.median(float(r["ccbs_s"]) for r in sub) * 1e3
        j_med = statistics.median(float(r["judgelight_s"]) for r in sub) * 1e3
        print(
            f"  {lo:>2}-{hi:<2} {len(sub):>6} "
            f"{c_ok:>6}/{len(sub):<2} {j_ok:>6}/{len(sub):<2} "
            f"{c_med:>8.1f}ms {j_med:>8.1f}ms"
        )


if __name__ == "__main__":
    main()
