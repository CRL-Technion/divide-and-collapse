"""Single source of truth for every full-suite number quoted in the paper.

Computes the Q1 aggregate fit, Q2 decomposition details, Q3 cost agreement,
and the CCBS ablation prose numbers from the *_full parquets, plus a couple
of derived quantities (H-build range, agent-share by family) that the
per-question summaries do not print. Q4 headline numbers come from
analyze_reeval.py --parquet reeval_full.parquet; this script reproduces the
aggregate table and head-to-head so everything lands in one place.

Run: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe \
        experiments/reeval/paper_numbers_full.py
"""

from __future__ import annotations

import statistics
from pathlib import Path

import polars as pl

RAW = Path(__file__).resolve().parents[2] / "results" / "raw"
BUCKETS = [(2, 4), (5, 9), (10, 19), (20, 29), (30, 39), (40, 50)]


def _parse_sizes(cell):
    if cell is None:
        return []
    s = str(cell).strip().strip("[]")
    return [int(x) for x in s.split(",") if x.strip()] if s else []


def pct(x):
    return sorted(x)


def _fit(ks, ts):
    n = len(ks)
    mk = sum(ks) / n
    mt = sum(ts) / n
    num = sum((k - mk) * (t - mt) for k, t in zip(ks, ts, strict=False))
    den = sum((k - mk) ** 2 for k in ks) or 1.0
    sl = num / den
    ic = mt - sl * mk
    pred = [ic + sl * k for k in ks]
    ssr = sum((t - p) ** 2 for t, p in zip(ts, pred, strict=False))
    sst = sum((t - mt) ** 2 for t in ts) or 1.0
    return sl, ic, 1 - ssr / sst


print("=" * 72)
print("Q1 aggregate (full)")
q = pl.read_parquet(RAW / "q1_pogema_timings_full.parquet")
us = [x * 1e6 for x in q["median_s"].to_list()]
print(f"  rows (timed solves): {q.height:,}")
print(
    f"  median {statistics.median(us):.2f}us  p99 {pct(us)[int(0.99*len(us))]:.1f}us  max {max(us):.0f}us"
)
ks = q["k_i"].to_list()
ts = q["median_s"].to_list()
sl, ic, r2 = _fit(ks, ts)
print(f"  AGGREGATE fit (all k_i): slope {sl*1e9:.0f}ns/step  offset {ic*1e6:.2f}us  R2 {r2:.3f}")
# per-family median us range
fam_meds = []
for fam in sorted(q["dataset"].unique().to_list()):
    s = q.filter(pl.col("dataset") == fam)
    fam_meds.append(statistics.median([x * 1e6 for x in s["median_s"].to_list()]))
print(f"  per-family medians range: {min(fam_meds):.1f}-{max(fam_meds):.1f}us")
tot_steps = sum(ks)
tot_s = sum(q["median_s"].to_list())
print(
    f"  throughput {tot_steps/tot_s/1e6:.2f} M-vertices/s ; {1/statistics.median(q['median_s'].to_list()):.0f} solves/s (median)"
)

print("=" * 72)
print("Q2 decomposition (full)")
d = pl.read_parquet(RAW / "q2_decomposition_full.parquet")
print(f"  instances {d.height}")
print(f"  mean singleton fraction {d['singleton_fraction'].mean()*100:.1f}%")
print(f"  overall median H-build {d['h_construction_time_s'].median()*1e3:.2f}ms")
# H-build range: by num_agents extremes
small = d.sort("num_agents").head(50)
big = d.sort("num_agents", descending=True).head(50)
print(
    f"  smallest-N median H-build {small['h_construction_time_s'].median()*1e3:.2f}ms (N~{small['num_agents'].median():.0f})"
)
print(
    f"  largest-N  median H-build {big['h_construction_time_s'].median()*1e3:.1f}ms (N~{big['num_agents'].median():.0f}); max N {d['num_agents'].max()}"
)
tot_single = d["n_singletons"].sum()
tot_nt = d["n_non_trivial"].sum()
print(
    f"  singleton component fraction {100*tot_single/(tot_single+tot_nt):.1f}%  ({tot_nt} non-trivial)"
)
# nt size distribution + agent share
comp = {f"[{lo},{hi}]": 0 for lo, hi in BUCKETS}
comp["[50+]"] = 0
agent = {k: 0 for k in comp}
nt_sizes = []
for row in d.iter_rows(named=True):
    for sz in _parse_sizes(row["non_trivial_sizes"]):
        nt_sizes.append(sz)
        b = (
            next((f"[{lo},{hi}]" for lo, hi in BUCKETS if lo <= sz <= hi), "[50+]")
            if sz <= 50
            else "[50+]"
        )
        comp[b] += 1
        agent[b] += sz
ntot = sum(comp.values())
atot = sum(agent.values())
print(
    f"  nt components {len(nt_sizes)}  median {statistics.median(nt_sizes):.0f}  p90 {pct(nt_sizes)[int(0.9*len(nt_sizes))]}  p99 {pct(nt_sizes)[int(0.99*len(nt_sizes))]}  max {max(nt_sizes)}"
)
for b in comp:
    print(f"    {b:>8}  %nt-comp {100*comp[b]/ntot:5.1f}  %agents {100*agent[b]/atot:5.1f}")
# per family table cells
print("  per-family (Mean N | Singleton% | Mean #non-triv | Max|I| | Median H-build):")
for fam in sorted(d["dataset"].unique().to_list()):
    s = d.filter(pl.col("dataset") == fam)
    meanN = s["num_agents"].mean()
    sgl = s["singleton_fraction"].mean() * 100
    mnt = s["n_non_trivial"].mean()
    maxI = s["max_non_trivial_size"].max()
    hb = s["h_construction_time_s"].median() * 1e3
    # agent-share in 50+ for this family
    a50 = atotf = 0
    for row in s.iter_rows(named=True):
        for sz in _parse_sizes(row["non_trivial_sizes"]):
            atotf += sz
            if sz >= 50:
                a50 += sz
    share = 100 * a50 / atotf if atotf else 0.0
    print(
        f"    {fam:<13} {meanN:6.1f} | {sgl:5.1f} | {mnt:5.2f} | {maxI:>4} | {hb:6.2f}ms | agents50+ {share:4.1f}%"
    )

print("=" * 72)
print("Q3 cost agreement (full, joint_comparison_full)")
j = pl.read_parquet(RAW / "joint_comparison_full.parquet")
nt_le50 = j.height
csolved = j.filter(pl.col("ccbs_success")).height
both = j.filter(pl.col("both_succeeded"))
matched = both.filter(pl.col("cost_match")).height
ccbs_better = sum(
    1
    for r in both.iter_rows(named=True)
    if r["ccbs_cost"] is not None
    and r["judgelight_cost"] is not None
    and r["ccbs_cost"] < r["judgelight_cost"]
)
jl_better = sum(
    1
    for r in both.iter_rows(named=True)
    if r["ccbs_cost"] is not None
    and r["judgelight_cost"] is not None
    and r["ccbs_cost"] > r["judgelight_cost"]
)
print(f"  components size<=50: {nt_le50}")
print(f"  CCBS solved {csolved}/{nt_le50} ({100*csolved/nt_le50:.1f}%)")
print(
    f"  both succeeded {both.height}; cost match {matched} ({100*matched/both.height:.1f}%); "
    f"CCBS strictly better {ccbs_better} ({100*ccbs_better/both.height:.1f}%); JL better {jl_better}"
)

print("=" * 72)
print("Ablation (full) solve rates")
a = pl.read_parquet(RAW / "ablation_lazylb_full.parquet")
for v in ["vanilla", "cardinal", "disjoint", "full"]:
    sv = a.filter(pl.col("variant") == v)
    rate = 100 * sv.filter(pl.col("solved")).height / sv.height
    print(f"  {v:>9} {rate:.1f}%  (n={sv.height})")
print(f"  total components per variant: {a.filter(pl.col('variant')=='vanilla').height}")
