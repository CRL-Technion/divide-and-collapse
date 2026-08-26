"""Generate the Q1 scaling figure from POGEMA timings.

Reads ``results/raw/q1_pogema_timings.parquet`` and writes
``results/figures/q1_scaling.pdf`` (and a PNG companion). One subplot per
POGEMA family: scatter of (k_i, median microseconds) with a linear fit and
the R^2 annotated in the panel title.

Sized for the full text width (~7 in, a two-column ``figure*``) so the
five panels and their ~9 pt fonts stay legible in the paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
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
FIG_OUT = REPO_ROOT / "results" / "figures" / "q1_scaling.pdf"

FAMILY_ORDER = ["01-random", "02-mazes", "03-warehouse", "04-movingai", "05-puzzles"]
FAMILY_LABEL = {
    "01-random": "Random",
    "02-mazes": "Maze",
    "03-warehouse": "Warehouse",
    "04-movingai": "MovingAI",
    "05-puzzles": "Puzzle",
}


def _fit(ks: list[int], ts: list[float]) -> tuple[float, float, float]:
    n = len(ks)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
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
    return slope, intercept, r2


def main() -> None:
    pog = pl.read_parquet(POGEMA_PARQUET)
    pog = pog.with_columns((pl.col("median_s") * 1e6).alias("us"))

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for ax, family in zip(axes_flat[:5], FAMILY_ORDER, strict=False):
        sub = pog.filter(pl.col("dataset") == family)
        if sub.height == 0:
            ax.set_title(f"{FAMILY_LABEL[family]} (no data)")
            ax.axis("off")
            continue
        ks = sub["k_i"].to_list()
        us = sub["us"].to_list()
        slope_s, intercept_s, r2 = _fit(ks, [u * 1e-6 for u in us])
        ax.scatter(ks, us, s=4, alpha=0.18, color="tab:blue", edgecolors="none")
        xs = sorted(set(ks))
        ys = [(intercept_s + slope_s * x) * 1e6 for x in xs]
        ax.plot(xs, ys, color="tab:red", linewidth=1.4)
        ax.set_title(f"{FAMILY_LABEL[family]}  ($R^2$={r2:.2f})")
        ax.grid(True, alpha=0.3)

    # Sixth cell is unused (five families); hide it.
    axes_flat[5].axis("off")

    fig.supxlabel(r"$k_i$ (schedule length)", fontsize=9)
    fig.supylabel(r"median runtime [$\mu$s]", fontsize=9)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150, bbox_inches="tight")
    fig.savefig(FIG_OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {FIG_OUT}")


if __name__ == "__main__":
    main()
