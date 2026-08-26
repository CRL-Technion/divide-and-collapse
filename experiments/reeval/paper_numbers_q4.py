"""Every Q4 / App-B.2 number the paper quotes, from a single campaign parquet.

Reads ``results/raw/reeval_arxiv.parquet`` -- one uniform campaign, seven
configurations over all 3,296 instances, five timing repeats per cell except
cells whose first two runs both fail -- and prints every quantity the paper
states for Q4 and App. B.2, so the tables can be checked against the data in
one command.

Run: .venv/Scripts/python.exe experiments/reeval/paper_numbers_q4.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

RAW = Path(__file__).resolve().parents[2] / "results" / "raw"
PARQUET = RAW / "reeval_arxiv.parquet"

# paper name -> config key
NAMES = [
    ("NoCollapse", "nocollapse"),
    ("IndepLB", "indep_lb"),
    ("Judgelight", "judgelight"),
    ("DnC+JL", "mapfc+judgelight"),
    ("DnC+C-CBS", "mapfc+ccbs"),
    ("DnC+Hyb (gate40)", "mapfc+ccbs+jl+gate40"),
    ("hybrid, no routing", "mapfc+ccbs+jl"),
]
HYB = "mapfc+ccbs+jl+gate40"
NOGATE = "mapfc+ccbs+jl"
JL = "judgelight"
KEY = ["dataset", "scenario", "num_agents"]


def ms(x: float) -> float:
    """Seconds -> milliseconds."""
    return 1e3 * x


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    df = pl.read_parquet(PARQUET)
    n_inst = df.filter(pl.col("config") == JL).height

    # ---------------------------------------------------------------- Tbl. 3
    rule(f"TABLE 3 -- aggregate over all {n_inst} instances (30 s limit)")
    print(
        f"{'configuration':<20} {'mean saving':>12} {'median time':>12} "
        f"{'success rate':>14} {'repeats':>10}"
    )
    for label, cfg in NAMES:
        sub = df.filter(pl.col("config") == cfg)
        solved = sub.filter(pl.col("isr"))
        sav = solved["saving_ratio"].mean()
        med = sub["runtime_total_s"].median()
        sr = 100.0 * solved.height / sub.height
        reps = f"{sub['timing_repeats'].min()}-{sub['timing_repeats'].max()}"
        print(
            f"{label:<20} {sav:>12.3f} {ms(med):>10.1f} ms "
            f"{sr:>8.1f}% ({solved.height}/{sub.height}) {reps:>8}"
        )

    # -------------------------------------------------- head-to-head vs JL
    rule("HEAD-TO-HEAD: DnC+Hyb vs Judgelight (cost)")
    h = df.filter(pl.col("config") == HYB).select([*KEY, "soc", "isr", "runtime_total_s"])
    j = df.filter(pl.col("config") == JL).select([*KEY, "soc", "isr", "runtime_total_s"])
    m = h.join(j, on=KEY, suffix="_jl")
    both = m.filter(pl.col("soc").is_not_null() & pl.col("soc_jl").is_not_null())
    win = both.filter(pl.col("soc") < pl.col("soc_jl")).height
    tie = both.filter(pl.col("soc") == pl.col("soc_jl")).height
    loss = both.filter(pl.col("soc") > pl.col("soc_jl")).height
    print(f"comparable instances (both returned a plan): {both.height}")
    print(
        f"hybrid strictly cheaper: {win} ({100 * win / both.height:.0f}%)   "
        f"ties: {tie} ({100 * tie / both.height:.0f}%)   costlier: {loss}"
    )
    both_isr = m.filter(pl.col("isr") & pl.col("isr_jl")).height
    print(f"  (both-solved subset, isr on both sides: {both_isr})")

    # ---------------------------------------------------------- speedups
    rule("SPEEDUPS: Judgelight / DnC+Hyb")
    sp = both.with_columns(
        (pl.col("runtime_total_s_jl") / pl.col("runtime_total_s")).alias("ratio")
    )
    med_jl = both["runtime_total_s_jl"].median()
    med_hyb = both["runtime_total_s"].median()
    print(f"median per-instance speedup : {sp['ratio'].median():.1f}x")
    print(
        f"ratio of median runtimes    : {med_jl / med_hyb:.1f}x  "
        f"({ms(med_jl):.1f} ms / {ms(med_hyb):.1f} ms)"
    )

    # ------------------------------------------- coordination-free (J2)
    rule("COORDINATION-FREE INSTANCES (all agents singletons)")
    # Judgelight does not decompose, so its nontrivial_count is 0 on every row;
    # identify coordination-free instances from a decomposing configuration.
    free = df.filter((pl.col("config") == HYB) & (pl.col("nontrivial_count") == 0)).select(KEY)
    print(f"coordination-free instances: {free.height} of {n_inst}")
    for label, cfg in [("IndepLB", "indep_lb"), ("DnC+Hyb", HYB)]:
        a = df.filter(pl.col("config") == cfg).select([*KEY, "runtime_total_s"])
        sub = free.join(a, on=KEY).join(
            df.filter(pl.col("config") == JL).select([*KEY, "runtime_total_s"]),
            on=KEY,
            suffix="_jl",
        )
        m_ours = sub["runtime_total_s"].median()
        m_jl = sub["runtime_total_s_jl"].median()
        print(
            f"  {label:<9} on those: median {ms(m_ours):.3f} ms vs "
            f"JL {ms(m_jl):.1f} ms  -> {m_jl / m_ours:.0f}x"
        )
    il = df.filter(pl.col("config") == "indep_lb")["runtime_total_s"].median()
    jl = df.filter(pl.col("config") == JL)["runtime_total_s"].median()
    print(f"  suite-wide med(JL)/med(IndepLB), for contrast: {jl / il:.0f}x")

    # ----------------------------------------------------------- Tbl. 4
    rule("TABLE 4 -- per-family (App. B.2)")
    print(
        f"{'family':<13} {'JL sav':>8} {'JL med':>10} {'Hyb sav':>8} "
        f"{'Hyb med':>10} {'NoRoute':>10} {'speedup':>9}"
    )
    for fam in sorted(df["dataset"].unique().to_list()):
        row = []
        for cfg in (JL, HYB, NOGATE):
            s = df.filter((pl.col("config") == cfg) & (pl.col("dataset") == fam))
            row.append(
                (s.filter(pl.col("isr"))["saving_ratio"].mean(), s["runtime_total_s"].median())
            )
        print(
            f"{fam:<13} {row[0][0]:>8.3f} {ms(row[0][1]):>8.1f} ms "
            f"{row[1][0]:>8.3f} {ms(row[1][1]):>8.1f} ms {ms(row[2][1]):>8.1f} ms "
            f"{row[0][1] / row[1][1]:>8.1f}x"
        )

    # ------------------------------------------------- saving deltas (J11)
    rule("Saving: gated vs un-gated hybrid, per family")
    for fam in sorted(df["dataset"].unique().to_list()):
        a = df.filter((pl.col("config") == HYB) & (pl.col("dataset") == fam))
        b = df.filter((pl.col("config") == NOGATE) & (pl.col("dataset") == fam))
        sa = a.filter(pl.col("isr"))["saving_ratio"].mean()
        sb = b.filter(pl.col("isr"))["saving_ratio"].mean()
        print(f"  {fam:<13} gate {sa:.6f}  no-routing {sb:.6f}  delta {abs(sa - sb):.2e}")

    # --------------------------------------------- cost stability (App B.2)
    rule("COST FLIPS ACROSS TIMING REPEATS")
    for label, cfg in NAMES:
        sub = df.filter(pl.col("config") == cfg)
        flips = sub.filter(~pl.col("cost_stable")).height
        print(f"  {label:<20} {flips} instance(s) whose cost varies across repeats")

    # ------------------------------------------------ JL budget note (J8)
    rule("Judgelight budget behaviour")
    jsub = df.filter(pl.col("config") == JL)
    unsolved = jsub.filter(~pl.col("isr"))
    print(f"JL instances not marked solved: {unsolved.height}")
    for r in unsolved.select([*KEY, "soc", "runtime_total_s", "failure_reason"]).to_dicts():
        print(
            f"   {r['dataset']}/{r['scenario']}/n{r['num_agents']}  "
            f"soc={r['soc']}  t={r['runtime_total_s']:.1f} s  "
            f"reason={r['failure_reason']}"
        )
    near = jsub.filter(pl.col("runtime_total_s") > 29.0).height
    print(f"JL solves whose median runtime exceeds 29 s: {near}")

    # ------------------------------------- regime-agnostic penalty (App B.2)
    rule("REGIME-AGNOSTIC VARIANT vs JUDGELIGHT, per family")
    for fam in sorted(df["dataset"].unique().to_list()):
        a = df.filter((pl.col("config") == NOGATE) & (pl.col("dataset") == fam))
        b = df.filter((pl.col("config") == JL) & (pl.col("dataset") == fam))
        r = a["runtime_total_s"].median() / b["runtime_total_s"].median()
        verdict = f"SLOWER than JL by {r:.1f}x" if r > 1 else f"{1 / r:.1f}x faster than JL"
        print(f"  {fam:<13} {verdict}")

    # ------------------------------------------------------- misc prose
    rule("MISC PROSE NUMBERS")
    ccbs = df.filter(pl.col("config") == "mapfc+ccbs")
    n_ok = ccbs.filter(pl.col("isr")).height
    print(f"DnC+C-CBS success plateau: {100 * n_ok / ccbs.height:.1f}% ({n_ok}/{ccbs.height})")
    hyb = df.filter(pl.col("config") == HYB)
    n_hyb = hyb.filter(pl.col("isr")).height
    print(f"DnC+Hyb success: {100 * n_hyb / hyb.height:.1f}% ({n_hyb}/{hyb.height})")


if __name__ == "__main__":
    main()
