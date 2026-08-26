"""Scatter + success-rate plots for CCBS vs Judgelight on POGEMA components.

Reads ``results/raw/joint_comparison.parquet`` and produces a 2-panel figure
at ``results/figures/joint_comparison.pdf`` (+ PNG companion):

- Left: log-log scatter of CCBS runtime vs Judgelight runtime for the
  components where both solvers succeeded inside the budget. y=x line and
  10x speed-up reference lines included.
- Right: per-component-size success rate for each solver as a stacked bar.
  Buckets: 2-4, 5-9, 10-19, 20-50.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]
PARQUET = REPO / "results" / "raw" / "joint_comparison.parquet"
FIG_PDF = REPO / "results" / "figures" / "joint_comparison.pdf"
FIG_PNG = FIG_PDF.with_suffix(".png")

FAMILY_COLOR = {
    "01-random": "tab:blue",
    "02-mazes": "tab:orange",
    "03-warehouse": "tab:green",
    "04-movingai": "tab:red",
    "05-puzzles": "tab:purple",
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=PARQUET)
    args = parser.parse_args()
    df = pl.read_parquet(args.parquet)

    both = df.filter(pl.col("both_succeeded"))

    fig, (ax_scatter, ax_success) = plt.subplots(1, 2, figsize=(14, 6))

    if both.height:
        ccbs_us = (both["ccbs_s"].to_numpy() * 1e6).clip(min=1.0)
        jl_us = (both["judgelight_s"].to_numpy() * 1e6).clip(min=1.0)
        families = both["dataset"].to_list()
        for fam in sorted({f for f in families}):
            mask = np.array([f == fam for f in families])
            ax_scatter.scatter(
                jl_us[mask],
                ccbs_us[mask],
                s=12,
                alpha=0.5,
                color=FAMILY_COLOR.get(fam, "k"),
                label=fam,
                edgecolors="none",
            )
        lo = float(min(ccbs_us.min(), jl_us.min())) * 0.5
        hi = float(max(ccbs_us.max(), jl_us.max())) * 2.0
        grid = np.array([lo, hi])
        ax_scatter.plot(grid, grid, color="k", linestyle="--", linewidth=0.8, label="y = x")
        ax_scatter.plot(
            grid, grid / 10, color="gray", linestyle=":", linewidth=0.6, label="CCBS 10x faster"
        )
        ax_scatter.plot(
            grid, grid * 10, color="gray", linestyle=":", linewidth=0.6, label="CCBS 10x slower"
        )
        ax_scatter.set_xscale("log")
        ax_scatter.set_yscale("log")
        ax_scatter.set_xlabel(r"Judgelight runtime [$\mu$s]")
        ax_scatter.set_ylabel(r"CCBS runtime [$\mu$s]")
        ax_scatter.set_title(f"Both-succeeded scatter ({both.height} components)", fontsize=11)
        ax_scatter.grid(True, which="both", alpha=0.3)
        ax_scatter.legend(loc="upper left", fontsize=8, framealpha=0.9)

    buckets = [(2, 4), (5, 9), (10, 19), (20, 50)]
    labels = [f"{lo}-{hi}" for lo, hi in buckets]
    ccbs_rates = []
    jl_rates = []
    counts = []
    for lo, hi in buckets:
        sub = df.filter((pl.col("comp_size") >= lo) & (pl.col("comp_size") <= hi))
        if sub.height == 0:
            ccbs_rates.append(0.0)
            jl_rates.append(0.0)
            counts.append(0)
            continue
        c_ok = int(sub["ccbs_success"].sum())
        j_ok = int(sub["judgelight_success"].sum())
        ccbs_rates.append(100 * c_ok / sub.height)
        jl_rates.append(100 * j_ok / sub.height)
        counts.append(sub.height)

    x = np.arange(len(labels))
    width = 0.35
    ax_success.bar(x - width / 2, ccbs_rates, width, label="CCBS", color="tab:blue", alpha=0.85)
    ax_success.bar(
        x + width / 2, jl_rates, width, label="Judgelight", color="tab:orange", alpha=0.85
    )
    for i, n in enumerate(counts):
        ax_success.text(i, 102, f"n={n}", ha="center", fontsize=8)
    ax_success.set_xticks(x)
    ax_success.set_xticklabels(labels)
    ax_success.set_xlabel("Component size")
    ax_success.set_ylabel("Success rate within budget (%)")
    ax_success.set_ylim(0, 115)
    ax_success.set_title("Success rate per component-size bucket", fontsize=11)
    ax_success.legend(loc="upper right", fontsize=9)
    ax_success.grid(True, axis="y", alpha=0.3)

    fig.suptitle("CCBS vs Judgelight on POGEMA non-trivial components", fontsize=12)
    fig.tight_layout()
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, dpi=150)
    fig.savefig(FIG_PNG, dpi=150)
    print(f"wrote {FIG_PDF}")
    print(f"wrote {FIG_PNG}")


if __name__ == "__main__":
    main()
