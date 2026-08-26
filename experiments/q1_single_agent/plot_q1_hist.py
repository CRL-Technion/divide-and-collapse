r"""Q1 figure: distribution of \oneMAPFC solve times.

Complements the Q1 scaling scatter (``plot_q1.py``, runtime vs $k_i$) by
showing the *marginal* runtime distribution: a histogram of the solve time
over every timed solve logged in
``results/raw/q1_pogema_timings_full.parquet`` (one entry per agent-prefix
pair, across the five POGEMA families). Writes
``results/figures/q1_runtime_hist.pdf`` (+ ``.png``), sized for a single
column (~3.3 in).

The x-axis is the per-agent solve time in microseconds; the y-axis is the
count of timed solves. Median and 99th-percentile runtimes are marked,
and the count, median, p99, and max are annotated. The point of the figure
is that the linear-time per-agent primitive resolves essentially every
schedule in tens of microseconds, with a thin right tail.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
POGEMA_PARQUET = REPO_ROOT / "results" / "raw" / "q1_pogema_timings_full.parquet"
FIG_OUT = REPO_ROOT / "results" / "figures" / "q1_runtime_hist.pdf"


def main() -> None:
    df = pl.read_parquet(POGEMA_PARQUET)
    us = np.array([t * 1e6 for t in df["median_s"].to_list()], dtype=float)
    us = us[us > 0.0]
    n = us.size
    med = float(np.median(us))
    p99 = float(np.percentile(us, 99))
    mx = float(us.max())

    # Cap the visible axis at 50 us: this covers the bulk and the 99.9th
    # percentile (~47 us); the handful of solves in the thin tail past 50 us
    # are dropped from the view but documented via the "max" annotation.
    x_cap = 50.0

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    bins = np.linspace(0.0, x_cap, 51)
    ax.hist(
        us[us <= x_cap],
        bins=bins,
        color="tab:blue",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xlim(0.0, x_cap)

    ax.axvline(
        med, color="tab:red", linestyle="--", linewidth=1.4, label=f"median ${med:.1f}\\,\\mu$s"
    )
    ax.axvline(
        p99, color="tab:orange", linestyle=":", linewidth=1.4, label=f"p99 ${p99:.1f}\\,\\mu$s"
    )

    ax.set_xlabel(r"Per-agent 1-MAPFC solve time [$\mu$s]")
    ax.set_ylabel("Number of timed solves")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    ax.annotate(
        f"$n={n:,}$ solves\nmed. ${med:.1f}\\,\\mu$s, p99 ${p99:.1f}\\,\\mu$s\nmax ${mx:.0f}\\,\\mu$s",
        xy=(0.97, 0.62),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "linewidth": 0.4},
    )

    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150, bbox_inches="tight")
    fig.savefig(FIG_OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {FIG_OUT}")


if __name__ == "__main__":
    main()
