"""Q4 cactus / runtime-distribution figure.

Replaces the budget-pinned ISR columns: instead of reporting the fraction of
instances solved at a fixed wall-clock budget (5 s / 30 s), we plot the whole
empirical runtime distribution. For each solver configuration the curve is
the cumulative fraction of the 500 POGEMA instances solved within a given
per-instance wall-clock budget; a configuration that never solves an instance
(within the 30 s cap that produced the data) plateaus below 1.0, and that
plateau height *is* the configuration's instance-success rate at the cap.

Reads ``results/raw/q4_end_to_end_30s_lazylb.parquet`` and writes
``results/figures/q4_cactus.pdf`` (+ ``.png``). Sized for a single column
(~3.3 in) so the ~9 pt fonts survive into the paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

mpl.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)

REPO = Path(__file__).resolve().parents[2]
PARQUET = REPO / "results" / "raw" / "reeval_arxiv.parquet"
FIG_DIR = REPO / "results" / "figures"

# Only the four real solver configurations; NoCollapse/IndepLB are trivial
# reference points (UB / LB), not solvers, so they are omitted.
CONFIGS = (
    ("judgelight", "tab:orange", "-", "Judgelight"),
    ("mapfc+judgelight", "tab:red", "-.", "DnC + JL"),
    ("mapfc+ccbs", "tab:green", "--", "DnC + C-CBS"),
    ("mapfc+ccbs+jl+gate40", "tab:purple", "-", "DnC + C-CBS→JL"),
)


def main() -> None:
    df = pl.read_parquet(PARQUET)
    n_total = df.filter(pl.col("config") == "judgelight").height  # 500

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for name, color, ls, label in CONFIGS:
        sub = df.filter((pl.col("config") == name) & pl.col("isr"))
        ts = np.array(sorted(r * 1e3 for r in sub["runtime_total_s"].to_list()))  # ms
        if ts.size == 0:
            continue
        ys = np.arange(1, ts.size + 1) / n_total
        # Prepend a point at the first runtime with y=0 so the step starts cleanly.
        xs = np.concatenate(([ts[0]], ts))
        ys = np.concatenate(([0.0], ys))
        ax.step(xs, ys, where="post", color=color, linestyle=ls, linewidth=1.4, label=label)

    for bx, blab in ((5_000, "5 s"), (30_000, "30 s")):
        ax.axvline(bx, color="gray", linestyle=":", linewidth=0.9, alpha=0.7)
        ax.text(bx, 1.03, blab, fontsize=8, color="gray", ha="center", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel(r"Per-instance wall-clock budget [ms]")
    ax.set_ylabel("Fraction of instances solved")
    ax.set_ylim(0, 1.02)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIG_DIR / "q4_cactus.pdf"
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
