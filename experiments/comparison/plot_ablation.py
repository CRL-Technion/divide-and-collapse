"""Two-panel CCBS engineering-ablation figure.

Reads ``results/raw/ablation_lazylb.parquet`` (produced by ``run_ablation.py``)
and writes ``results/figures/q3_ablation.pdf`` (+ ``.png``):

- Left:  median high-level nodes expanded (``pop_count``) by component-size
         bucket, one curve per variant in the incremental ladder. Node count is
         deterministic (unlike wall-clock, which is confounded by the per-node
         cardinal-classification cost and by timeouts at the per-component cap),
         and is the standard CBS-ablation metric.
- Right: solve rate within the 5 s per-component budget, by size.

Sized for the full text width (~7 in, a two-column ``figure*``) so the two
panels and their ~9 pt fonts stay legible in the paper.

The story the figure carries: vanilla CCBS's search tree explodes on the large
coupled cores (right panel, its solve rate craters with size); cardinal-conflict
selection shrinks the tree by an order of magnitude, disjoint splitting reaches
~99.7% coverage, and conflict bypass trims the largest bucket further.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import polars as pl

mpl.rcParams.update(
    {
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
    }
)

REPO = Path(__file__).resolve().parents[2]
PARQUET = REPO / "results" / "raw" / "ablation_lazylb_full.parquet"
FIG_DIR = REPO / "results" / "figures"

BUCKETS = [(2, 4), (5, 9), (10, 19), (20, 29), (30, 39), (40, 50)]
BUCKET_LABELS = [f"{lo}–{hi}" for lo, hi in BUCKETS]  # noqa: RUF001 (en dash is deliberate in axis labels)

# (variant key, display label, color, linestyle, marker)
VARIANTS = [
    ("vanilla", "vanilla", "tab:gray", "-", "o"),
    ("cardinal", "+cardinal", "tab:blue", "--", "s"),
    ("disjoint", "+disjoint split", "tab:orange", "-.", "D"),
    ("full", "+bypass (full)", "tab:red", "-", "v"),
]


def _bucket_index(size: int) -> int | None:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= size <= hi:
            return i
    return None


def main() -> None:
    df = pl.read_parquet(PARQUET)
    df = df.with_columns(
        pl.col("comp_size").map_elements(_bucket_index, return_dtype=pl.Int64).alias("bucket")
    ).filter(pl.col("bucket").is_not_null())

    fig, (ax_n, ax_s) = plt.subplots(1, 2, figsize=(3.4, 2.0))
    xs = list(range(len(BUCKETS)))

    for key, label, color, ls, marker in VARIANTS:
        sub = df.filter(pl.col("variant") == key)
        med_pops, solve_pct = [], []
        for bi in xs:
            cell = sub.filter(pl.col("bucket") == bi)
            if cell.height == 0:
                med_pops.append(float("nan"))
                solve_pct.append(float("nan"))
                continue
            med_pops.append(float(cell["pop_count"].median()))
            solve_pct.append(100.0 * float(cell["solved"].mean()))
        ax_n.plot(xs, med_pops, color=color, linestyle=ls, marker=marker, ms=4, lw=1.5, label=label)
        ax_s.plot(
            xs, solve_pct, color=color, linestyle=ls, marker=marker, ms=4, lw=1.5, label=label
        )

    ax_n.set_yscale("log")
    ax_n.set_xlabel(r"Component size $|I_\ell|$")
    ax_n.set_ylabel("Median nodes expanded")
    ax_n.set_xticks(xs)
    ax_n.set_xticklabels(BUCKET_LABELS, rotation=40, ha="right")
    ax_n.grid(True, which="both", alpha=0.3)

    ax_s.set_xlabel(r"Component size $|I_\ell|$")
    ax_s.set_ylabel("Success rate [%]")
    ax_s.set_xticks(xs)
    ax_s.set_xticklabels(BUCKET_LABELS, rotation=40, ha="right")
    ax_s.set_ylim(min(80, ax_s.get_ylim()[0]), 101)
    ax_s.grid(True, alpha=0.3)

    handles, labels = ax_n.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=True,
        framealpha=0.9,
        borderaxespad=0.2,
        columnspacing=0.8,
        handlelength=1.6,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "q3_ablation.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

    # Console summary so the prose numbers can be verified.
    print("\n## Median high-level nodes expanded by variant x size bucket")
    print("  " + "variant".rjust(9) + "".join(f"{f'{lo}-{hi}':>9}" for lo, hi in BUCKETS))
    for key, _label, *_ in VARIANTS:
        sub = df.filter(pl.col("variant") == key)
        cells = []
        for bi in xs:
            cell = sub.filter(pl.col("bucket") == bi)
            cells.append(f"{int(cell['pop_count'].median()):>8} " if cell.height else f"{'-':>8} ")
        print("  " + key.rjust(9) + "".join(cells))
    print("\n## Solve rate by variant (overall, size <= 50)")
    for key, _label, *_ in VARIANTS:
        sub = df.filter(pl.col("variant") == key)
        print(f"  {label:>16}: {100.0 * float(sub['solved'].mean()):.1f}%")


if __name__ == "__main__":
    main()
