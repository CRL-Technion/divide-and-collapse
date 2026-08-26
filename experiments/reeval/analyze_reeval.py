"""Post-pilot analysis of the median-of-k re-evaluation.

Loads results/raw/reeval_<scope>.parquet and reports:
  1. Integrity   -- cell counts, soc=None failures per config, cost_stable flips.
  2. Aggregate   -- per-config mean saving, median total/solve wall-clock (the
                    median-of-k), and ceiling (fraction solved within the cap),
                    with the cross-instance IQR on total runtime.
  3. Head-to-head -- hybrid vs Judgelight: median pairwise speedup (+ IQR and a
                    bootstrap 95% CI), ratio of medians, and cost wins/ties/losses.
  4. Compare     -- side-by-side against the numbers currently in the paper
                    (the single-run 500-subset Tbl. q4-aggregate-30s).

Usage: .venv/Scripts/python.exe experiments/reeval/analyze_reeval.py --scope 500
"""

from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]

# Current paper Tbl. q4-aggregate-30s (single-run 500-subset): mean saving, median ms, ceiling %.
PAPER = {
    "nocollapse": (0.000, 0.0, 100.0),
    "indep_lb": (0.379, 1.8, 100.0),
    "judgelight": (0.344, 523.1, 99.6),
    "mapfc+judgelight": (0.344, 541.8, 99.8),
    "mapfc+ccbs": (0.350, 85.6, 83.4),
    "mapfc+ccbs+jl": (0.345, 90.7, 100.0),
}
ORDER = ["nocollapse", "indep_lb", "judgelight", "mapfc+judgelight", "mapfc+ccbs", "mapfc+ccbs+jl"]


def _median(xs):
    return statistics.median(xs) if xs else float("nan")


def _quart(xs):
    if len(xs) < 2:
        v = xs[0] if xs else float("nan")
        return v, v
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return q[0], q[2]


def _bootstrap_median_ci(xs, *, iters=2000, seed=12345):
    if len(xs) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(xs)
    meds = []
    for _ in range(iters):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        meds.append(statistics.median(sample))
    meds.sort()
    lo = meds[int(0.025 * iters)]
    hi = meds[int(0.975 * iters)]
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scope", default="500")
    ap.add_argument("--parquet", type=Path, default=None)
    args = ap.parse_args()
    pq = args.parquet or (REPO / "results" / "raw" / f"reeval_{args.scope}.parquet")
    df = pl.read_parquet(pq)
    print(f"loaded {df.height} rows from {pq.name}\n")

    by = {c: df.filter(pl.col("config") == c) for c in df["config"].unique().to_list()}

    # ---- 1. integrity -----------------------------------------------------
    print("## 1. Integrity")
    print(f"{'config':<18} {'n':>4} {'solved':>7} {'fail':>5} {'cost_unstable':>14}")
    for c in ORDER:
        if c not in by:
            continue
        sub = by[c]
        n = sub.height
        solved = sub.filter(pl.col("soc").is_not_null()).height
        fail = n - solved
        unstable = sub.filter(~pl.col("cost_stable")).height if "cost_stable" in sub.columns else 0
        print(f"{c:<18} {n:>4} {solved:>7} {fail:>5} {unstable:>14}")
    print()

    # ---- 2. aggregate -----------------------------------------------------
    print("## 2. Aggregate (median-of-k)")
    print(
        f"{'config':<18} {'mean_sav':>9} {'med_total_ms':>13} {'IQR_total_ms':>22} "
        f"{'med_solve_ms':>13} {'ceiling':>8}"
    )
    agg = {}
    for c in ORDER:
        if c not in by:
            continue
        sub = by[c]
        n = sub.height
        savs = sub.filter(pl.col("saving_ratio").is_not_null())["saving_ratio"].to_list()
        totals = [x * 1e3 for x in sub["runtime_total_s"].to_list()]
        solves = [x * 1e3 for x in sub["runtime_solver_only_s"].to_list()]
        ceiling = 100.0 * sub.filter(pl.col("soc").is_not_null()).height / n
        mean_sav = sum(savs) / len(savs) if savs else float("nan")
        mt = _median(totals)
        p25, p75 = _quart(totals)
        agg[c] = (mean_sav, mt, ceiling)
        print(
            f"{c:<18} {mean_sav:>9.3f} {mt:>13.1f} {f'[{p25:.1f}, {p75:.1f}]':>22} "
            f"{_median(solves):>13.1f} {ceiling:>7.1f}%"
        )
    print()

    # ---- 3. head-to-head: hybrid vs judgelight ----------------------------
    print("## 3. Head-to-head: mapfc+ccbs+jl vs judgelight")
    jl = {r["instance_id"]: r for r in by["judgelight"].to_dicts()}
    hy = {r["instance_id"]: r for r in by["mapfc+ccbs+jl"].to_dicts()}
    common = [i for i in jl if i in hy and jl[i]["soc"] is not None and hy[i]["soc"] is not None]
    ratios = [
        jl[i]["runtime_total_s"] / hy[i]["runtime_total_s"]
        for i in common
        if hy[i]["runtime_total_s"] > 0
    ]
    med_pair = _median(ratios)
    rp25, rp75 = _quart(ratios)
    lo, hi = _bootstrap_median_ci(ratios)
    print(f"  instances compared: {len(common)}")
    print(
        f"  median pairwise speedup (JL/hybrid): {med_pair:.2f}x  "
        f"IQR [{rp25:.2f}, {rp75:.2f}]  boot95% [{lo:.2f}, {hi:.2f}]"
    )
    mj = [x * 1e3 for x in by["judgelight"]["runtime_total_s"].to_list()]
    mh = [x * 1e3 for x in by["mapfc+ccbs+jl"]["runtime_total_s"].to_list()]
    print(
        f"  ratio of medians: {_median(mj) / _median(mh):.2f}x  "
        f"(JL {_median(mj):.1f}ms / hybrid {_median(mh):.1f}ms)"
    )
    wins = sum(1 for i in common if hy[i]["soc"] < jl[i]["soc"])
    ties = sum(1 for i in common if hy[i]["soc"] == jl[i]["soc"])
    loss = sum(1 for i in common if hy[i]["soc"] > jl[i]["soc"])
    print(f"  cost vs JL (both solved, n={len(common)}): wins {wins}  ties {ties}  losses {loss}")
    # coordination-free speed: indep_lb vs judgelight median
    mi = _median([x * 1e3 for x in by["indep_lb"]["runtime_total_s"].to_list()])
    print(
        f"  IndepLB median {mi:.2f}ms -> {_median(mj) / mi:.0f}x faster than JL "
        f"(coordination-free limit)"
    )
    print()

    # ---- 4. compare vs current paper table --------------------------------
    print("## 4. New (median-of-k) vs current paper Tbl. q4-aggregate-30s")
    print(
        f"{'config':<18} {'sav_new':>8} {'sav_pap':>8}   {'med_new':>9} {'med_pap':>9}   "
        f"{'ceil_new':>9} {'ceil_pap':>9}"
    )
    for c in ORDER:
        if c not in agg:
            continue
        sn, mn, cn = agg[c]
        sp, mp, cp = PAPER[c]
        print(f"{c:<18} {sn:>8.3f} {sp:>8.3f}   {mn:>8.1f}m {mp:>8.1f}m   {cn:>8.1f}% {cp:>8.1f}%")


if __name__ == "__main__":
    main()
