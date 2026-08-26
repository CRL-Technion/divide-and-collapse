"""Q2 figures: per-family distribution of component sizes |I_l|.

Two complementary views of the same decomposition over the 500 POGEMA
instances in ``results/raw/q2_decomposition_500.parquet``:

- ``q2_component_size_hist.pdf`` -- COMPONENT-weighted: the fraction of each
  family's components that fall in each size bucket, including the singleton
  bucket ``[1]``. Singletons dominate the component count (~88%), so the
  y-axis is logarithmic to keep the small non-trivial buckets legible. This
  is the figure that backs the small-component hypothesis (most components
  are singletons or tiny).

- ``q2_agent_size_dist.pdf`` -- AGENT-weighted: the fraction of each family's
  agents that sit in a component of each size bucket. This is the
  complementary lens: it shows that, although components are mostly small,
  the few large cores hold a large share of the agents on the densest
  families, which is why those families drive the framework's runtime.

Both use the size buckets of Table 2 ([20,29]/[30,39]/[40,49]) plus the
singleton bucket [1] and the open-ended [50+] tail. Sized for the full text
width (~7 in, a two-column ``figure*``).
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
        "xtick.labelsize": 7,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)

REPO = Path(__file__).resolve().parents[2]
PARQUET = REPO / "results" / "raw" / "q2_decomposition_500.parquet"
FIG_DIR = REPO / "results" / "figures"

FAMILY_ORDER = ("01-random", "02-mazes", "03-warehouse", "04-movingai", "05-puzzles")
FAMILY_COLOR = {
    "01-random": "tab:blue",
    "02-mazes": "tab:orange",
    "03-warehouse": "tab:green",
    "04-movingai": "tab:red",
    "05-puzzles": "tab:purple",
}

# [1] singleton bucket + the Table-2 resolution on the residue + the [50+] tail.
BUCKETS = ((1, 1), (2, 4), (5, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 10**9))
BUCKET_LABELS = ("1", "2-4", "5-9", "10-19", "20-29", "30-39", "40-49", "50+")


def _parse_sizes(cell: str | None) -> list[int]:
    if cell is None:
        return []
    s = str(cell).strip().strip("[]")
    return [int(x) for x in s.split(",") if x.strip()] if s else []


def _family_sizes(df: pl.DataFrame, fam: str) -> list[int]:
    """Every component size for a family: n_singletons ones plus the non-trivial sizes."""
    sub = df.filter(pl.col("dataset") == fam)
    sizes: list[int] = []
    for row in sub.iter_rows(named=True):
        sizes.extend([1] * int(row["n_singletons"]))
        sizes.extend(_parse_sizes(row["non_trivial_sizes"]))
    return sizes


def _bucket_of(sz: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= sz <= hi:
            return i
    return -1


def _hist(
    df: pl.DataFrame, *, agent_weighted: bool, out_name: str, ylabel: str, ylog: bool
) -> None:
    families = [f for f in FAMILY_ORDER if f in set(df["dataset"].unique())]
    fig, axes = plt.subplots(1, len(families), figsize=(7.0, 2.3), sharey=True)
    if len(families) == 1:
        axes = [axes]

    x = np.arange(len(BUCKETS))
    for ax, fam in zip(axes, families, strict=False):
        sizes = _family_sizes(df, fam)
        weights = [0.0] * len(BUCKETS)
        for sz in sizes:
            b = _bucket_of(sz)
            if b >= 0:
                weights[b] += sz if agent_weighted else 1.0
        total = sum(weights) or 1.0
        fracs = [w / total for w in weights]
        # On a log axis a zero-height bar cannot be drawn; mask it to NaN.
        heights = [f if (f > 0 or not ylog) else np.nan for f in fracs]
        ax.bar(
            x,
            heights,
            color=FAMILY_COLOR.get(fam, "tab:gray"),
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_title(fam)
        ax.set_xticks(x)
        ax.set_xticklabels(BUCKET_LABELS, rotation=45, ha="right")
        if ylog:
            ax.set_yscale("log")
            ax.set_ylim(8e-4, 1.3)
        else:
            ax.set_ylim(0, 1.0)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel(ylabel)
    fig.supxlabel("Component size $|I_\\ell|$", fontsize=9)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIG_DIR / out_name
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


def _print_aggregate(df: pl.DataFrame) -> None:
    comp = [0] * len(BUCKETS)
    agent = [0] * len(BUCKETS)
    for fam in FAMILY_ORDER:
        for sz in _family_sizes(df, fam):
            b = _bucket_of(sz)
            if b >= 0:
                comp[b] += 1
                agent[b] += sz
    tc, ta = sum(comp) or 1, sum(agent) or 1
    nontriv = sum(comp[1:]) or 1
    print("\n## Aggregate over 500 instances")
    print(f"  components total {tc}, agents total {ta}, non-trivial components {nontriv}")
    print(f"  {'bucket':>8} {'%comp(all)':>11} {'%comp(non-triv)':>16} {'%agents':>9}")
    for i, lab in enumerate(BUCKET_LABELS):
        ntf = f"{100*comp[i]/nontriv:5.1f}%" if i >= 1 else "   --"
        print(f"  {lab:>8} {100*comp[i]/tc:10.1f}% {ntf:>16} {100*agent[i]/ta:8.1f}%")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=PARQUET)
    args = parser.parse_args()

    df = pl.read_parquet(args.parquet)
    print(f"loaded {df.height} instances from {args.parquet}")
    _hist(
        df,
        agent_weighted=False,
        out_name="q2_component_size_hist.pdf",
        ylabel="Fraction of\ncomponents (log)",
        ylog=True,
    )
    _hist(
        df,
        agent_weighted=True,
        out_name="q2_agent_size_dist.pdf",
        ylabel="Fraction of agents",
        ylog=False,
    )
    _print_aggregate(df)


if __name__ == "__main__":
    main()
