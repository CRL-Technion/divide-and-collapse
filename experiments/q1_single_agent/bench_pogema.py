"""POGEMA-derived Q1 benchmark: time solve_unconstrained on real schedules.

Consumes the JSONL written by ``scripts/dump_pogema_schedules.py``. For each
agent's trajectory ``M^i``, runs :func:`solve_unconstrained` ``REPEATS`` times
(with a small warmup) and records the median wall-clock together with
``k_i``, ``cost(M^i)``, and the optimal saving ``cost(M^i) - OPT({i})``.

Outputs a parquet file at ``results/raw/q1_pogema_timings.parquet`` with one
row per agent, plus a markdown summary on stdout: per-family linear fit
(slope, intercept, R^2) and a per-bin throughput in million-vertices-per-second.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import polars as pl

from mapfc.single_agent.collapse_dag import baseline_cost, solve_unconstrained

REPO_ROOT = Path(__file__).resolve().parents[2]
JSONL_IN = REPO_ROOT / "benchmarks" / "schedules" / "pogema_full.jsonl"
PARQUET_OUT = REPO_ROOT / "results" / "raw" / "q1_pogema_timings_full.parquet"

REPEATS = 11
WARMUP = 2
PREFIX_LENGTHS = (8, 16, 32, 48, 64, 96, 128)


def _load_agents() -> list[dict[str, object]]:
    if not JSONL_IN.exists():
        raise FileNotFoundError(f"{JSONL_IN} not found; run scripts/dump_pogema_schedules.py first")
    with JSONL_IN.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _time_solve(schedule: list[tuple[int, int]]) -> tuple[int, float]:
    for _ in range(WARMUP):
        solve_unconstrained(schedule)
    timings: list[float] = []
    cost = -1
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        cost, _ = solve_unconstrained(schedule)
        timings.append(time.perf_counter() - t0)
    return cost, statistics.median(timings)


def main() -> None:
    agents = _load_agents()
    rows: list[dict[str, object]] = []
    for entry in agents:
        traj = [tuple(xy) for xy in entry["trajectory"]]
        if not traj:
            continue
        for prefix_k in PREFIX_LENGTHS:
            if prefix_k + 1 > len(traj):
                continue
            sub = traj[: prefix_k + 1]
            opt_cost, elapsed_s = _time_solve(sub)
            baseline = baseline_cost(sub)
            rows.append(
                {
                    "dataset": entry["dataset"],
                    "scenario": entry["scenario"],
                    "num_agents": entry["num_agents"],
                    "agent_idx": entry["agent_idx"],
                    "k_i": prefix_k,
                    "baseline_cost": baseline,
                    "opt_cost": opt_cost,
                    "saving": baseline - opt_cost,
                    "median_s": elapsed_s,
                }
            )

    PARQUET_OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows)
    df.write_parquet(PARQUET_OUT)
    print(f"wrote {PARQUET_OUT} ({len(rows)} rows)")

    print("\n## Per-family Q1 fit")
    print(
        f"{'family':>14} {'agents':>8} {'k_med':>8} {'k_max':>8} "
        f"{'us_med':>10} {'slope':>14} {'R2':>8}"
    )
    summary_rows: list[dict[str, object]] = []
    for fam in sorted({entry["dataset"] for entry in agents}):
        sub = [r for r in rows if r["dataset"] == fam]
        if len(sub) < 4:
            continue
        ks = [r["k_i"] for r in sub]
        ts = [r["median_s"] for r in sub]
        n = len(ks)
        mean_k = sum(ks) / n
        mean_t = sum(ts) / n
        num = sum((k - mean_k) * (t - mean_t) for k, t in zip(ks, ts, strict=False))
        den = sum((k - mean_k) ** 2 for k in ks) or 1.0
        slope = num / den
        intercept = mean_t - slope * mean_k
        pred = [intercept + slope * k for k in ks]
        ss_res = sum((t - p) ** 2 for t, p in zip(ts, pred, strict=False))
        ss_tot = sum((t - mean_t) ** 2 for t in ts) or 1.0
        r2 = 1.0 - ss_res / ss_tot
        k_med = statistics.median(ks)
        k_max = max(ks)
        t_med_us = statistics.median(ts) * 1e6
        print(
            f"{fam:>14} {n:>8} {k_med:>8.0f} {k_max:>8.0f} "
            f"{t_med_us:>10.2f} {slope:>14.3e} {r2:>8.4f}"
        )
        summary_rows.append({"family": fam, "slope_s_per_step": slope, "r2": r2})

    print("\n## Aggregate")
    all_ks = [r["k_i"] for r in rows]
    all_ts = [r["median_s"] for r in rows]
    total_steps = sum(all_ks)
    total_secs = sum(all_ts)
    if total_secs > 0:
        print(
            f"aggregate throughput: {total_steps / total_secs / 1e6:.2f} " f"M-vertices-per-second"
        )
    print(f"total agents timed: {len(rows)}")
    print(f"total steps measured: {total_steps:,}")


if __name__ == "__main__":
    main()
