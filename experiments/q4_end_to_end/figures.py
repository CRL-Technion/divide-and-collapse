"""Q4 figures generated from ``results/raw/q4_end_to_end.parquet``.

Two figures land in ``results/figures/``:

- ``q4_saving_vs_runtime.pdf`` — Pareto-style scatter of mean saving
  ratio vs median solver-only runtime per configuration. The framework
  configurations (``mapfc+ccbs``, ``mapfc+judgelight``) should sit on the
  efficient frontier between monolithic Judgelight and trivial
  baselines.
- ``q4_saving_per_family.pdf`` — grouped bar chart of mean saving
  ratio per (family, config). Validates that the framework's headline
  numbers hold across all five POGEMA families.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]
PARQUET = REPO / "results" / "raw" / "q4_end_to_end.parquet"
FIG_DIR = REPO / "results" / "figures"

CONFIG_ORDER = (
    "nocollapse",
    "indep_lb",
    "judgelight",
    "mapfc+judgelight",
    "mapfc+ccbs",
    "mapfc+ccbs+jl",
)
CONFIG_COLOR = {
    "nocollapse": "tab:gray",
    "indep_lb": "tab:olive",
    "judgelight": "tab:orange",
    "mapfc+judgelight": "tab:red",
    "mapfc+ccbs": "tab:green",
    "mapfc+ccbs+jl": "tab:purple",
}
CONFIG_MARKER = {
    "nocollapse": "x",
    "indep_lb": "s",
    "judgelight": "o",
    "mapfc+judgelight": "D",
    "mapfc+ccbs": "*",
    "mapfc+ccbs+jl": "P",
}


def _saving_vs_runtime(df: pl.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in CONFIG_ORDER:
        sub = df.filter((pl.col("config") == name) & pl.col("isr"))
        if sub.height == 0:
            continue
        sav = float(sub["saving_ratio"].mean() or 0.0)
        rt = float(sub["runtime_solver_only_s"].median()) * 1e3
        ax.scatter(
            rt,
            sav,
            s=180 if name == "mapfc+ccbs" else 110,
            color=CONFIG_COLOR.get(name, "k"),
            marker=CONFIG_MARKER.get(name, "o"),
            label=name,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.annotate(
            name,
            (rt, sav),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Median solver-only runtime per instance [ms]")
    ax.set_ylabel("Mean SoC saving ratio (ISR instances only)")
    ax.set_title("Q4: saving vs runtime per configuration (lower-right is better)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_pdf = FIG_DIR / f"q4_saving_vs_runtime{_SUFFIX}.pdf"
    out_png = out_pdf.with_suffix(".png")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=150)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_pdf}")


def _saving_per_family(df: pl.DataFrame) -> None:
    families = sorted({str(d) for d in df["dataset"].unique()})
    config_names = [c for c in CONFIG_ORDER if c in df["config"].unique()]
    per: dict[tuple[str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in df.iter_rows(named=True):
        if not row["isr"] or row["saving_ratio"] is None:
            continue
        key = (row["dataset"], row["config"])
        per[key] += float(row["saving_ratio"])
        counts[key] += 1
    means = {k: per[k] / counts[k] if counts[k] else 0.0 for k in per}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(families))
    width = 0.18
    for i, cname in enumerate(config_names):
        offset = (i - (len(config_names) - 1) / 2) * width
        vals = [means.get((fam, cname), 0.0) for fam in families]
        ax.bar(
            x + offset,
            vals,
            width,
            label=cname,
            color=CONFIG_COLOR.get(cname, "k"),
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=10, ha="right")
    ax.set_xlabel("POGEMA family")
    ax.set_ylabel("Mean SoC saving ratio (ISR instances)")
    ax.set_title("Q4: mean SoC saving per family per configuration")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    fig.tight_layout()
    out_pdf = FIG_DIR / f"q4_saving_per_family{_SUFFIX}.pdf"
    out_png = out_pdf.with_suffix(".png")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=150)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_pdf}")


_SUFFIX = ""


def main() -> None:
    import argparse

    global _SUFFIX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=PARQUET)
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()
    _SUFFIX = args.suffix

    df = pl.read_parquet(args.parquet)
    print(f"loaded {df.height} rows ({df['config'].n_unique()} configs) from {args.parquet}")
    _saving_vs_runtime(df)
    _saving_per_family(df)


if __name__ == "__main__":
    main()
