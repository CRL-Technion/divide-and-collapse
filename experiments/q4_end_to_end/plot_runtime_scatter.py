"""Runtime-scatter plots: monolithic Judgelight vs MAPFC pipeline configs.

For each instance where both configs succeed (ISR), one point is drawn at
``(mapfc+ccbs runtime, judgelight runtime)``. Points are coloured by
POGEMA family. The y = x diagonal, the y = 10x line (CCBS 10x faster),
and the y = x/10 line (CCBS 10x slower) are included for orientation.

Produces:

- ``q4_runtime_ccbs_vs_jl.{pdf,png}`` — pure MAPFC+CCBS vs monolithic JL
- ``q4_runtime_ccbsjl_vs_jl.{pdf,png}`` — hybrid MAPFC+CCBS→JL vs monolithic JL
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]
PARQUET = REPO / "results" / "raw" / "q4_end_to_end.parquet"
FIG_DIR = REPO / "results" / "figures"

FAMILY_COLOR = {
    "01-random": "tab:blue",
    "02-mazes": "tab:orange",
    "03-warehouse": "tab:green",
    "04-movingai": "tab:red",
    "05-puzzles": "tab:purple",
}


def _scatter(
    df: pl.DataFrame,
    x_config: str,
    y_config: str,
    title: str,
    out_stem: str,
) -> None:
    x_rows = df.filter(pl.col("config") == x_config).sort("instance_id")
    y_rows = df.filter(pl.col("config") == y_config).sort("instance_id")
    both = x_rows.select(
        "instance_id",
        "dataset",
        pl.col("runtime_total_s").alias("x_s"),
        pl.col("isr").alias("x_isr"),
    ).join(
        y_rows.select(
            "instance_id",
            pl.col("runtime_total_s").alias("y_s"),
            pl.col("isr").alias("y_isr"),
        ),
        on="instance_id",
    )
    ok = both.filter(pl.col("x_isr") & pl.col("y_isr"))

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    families = sorted({str(d) for d in ok["dataset"].unique()})
    n_by_fam = {}
    for fam in families:
        sub = ok.filter(pl.col("dataset") == fam)
        n_by_fam[fam] = sub.height
        if sub.height == 0:
            continue
        xs = (sub["x_s"].to_numpy() * 1e3).clip(min=0.05)
        ys = (sub["y_s"].to_numpy() * 1e3).clip(min=0.05)
        ax.scatter(
            xs,
            ys,
            s=18,
            color=FAMILY_COLOR.get(fam, "k"),
            edgecolors="none",
            alpha=0.65,
            label=f"{fam} (n={sub.height})",
        )

    # Diagonal references
    lo = 0.05
    hi = (
        max(
            float((ok["x_s"].max() or 1.0) * 1e3),
            float((ok["y_s"].max() or 1.0) * 1e3),
        )
        * 1.4
    )
    grid = np.array([lo, hi])
    ax.plot(grid, grid, color="k", linestyle="--", linewidth=0.8, label="y = x")
    ax.plot(grid, grid * 10, color="gray", linestyle=":", linewidth=0.6, label="JL 10x slower")
    ax.plot(grid, grid / 10, color="gray", linestyle=":", linewidth=0.6, label="JL 10x faster")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"{x_config} runtime per instance [ms]")
    ax.set_ylabel(f"{y_config} runtime per instance [ms]")
    ax.set_title(f"{title}\nn = {ok.height} instances where both succeed")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out_pdf = FIG_DIR / f"{out_stem}.pdf"
    out_png = out_pdf.with_suffix(".png")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=150)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    plt.close(fig)


def main() -> None:
    df = pl.read_parquet(PARQUET)
    _scatter(
        df,
        x_config="mapfc+ccbs",
        y_config="judgelight",
        title="Monolithic Judgelight vs MAPFC+CCBS — per-instance runtime",
        out_stem="q4_runtime_ccbs_vs_jl",
    )
    _scatter(
        df,
        x_config="mapfc+ccbs+jl",
        y_config="judgelight",
        title="Monolithic Judgelight vs hybrid MAPFC+CCBS→JL — per-instance runtime",
        out_stem="q4_runtime_ccbsjl_vs_jl",
    )


if __name__ == "__main__":
    main()
