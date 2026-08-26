"""Q4 LaTeX tables generated from ``results/raw/q4_end_to_end.parquet``.

Two tables land in ``results/tables/``:

- ``q4_aggregate.tex`` — per-config success rate (ISR), mean saving
  ratio, median wall-clock split (total / solver-only / build).
  Designed to be ``\\input{...}``-ed into experiments.tex's headline
  comparison block.
- ``q4_per_family.tex`` — same metrics broken down by POGEMA family.

All tables use ``\\sisetup`` units so the numbers line up against
``experiments.tex``'s existing typography. No fabricated cells: a config
that returned no successful rows for a family writes ``--``.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]
PARQUET = REPO / "results" / "raw" / "q4_end_to_end.parquet"
TABLES_DIR = REPO / "results" / "tables"

CONFIG_ORDER = (
    "nocollapse",
    "indep_lb",
    "judgelight",
    "mapfc+judgelight",
    "mapfc+ccbs",
    "mapfc+ccbs+jl",
)
CONFIG_DISPLAY = {
    "nocollapse": r"\textsc{NoCollapse}",
    "indep_lb": r"\textsc{IndepLB}",
    "judgelight": r"\textsf{Judgelight}",
    "mapfc+judgelight": r"\textsc{MAPFC}+\textsf{JL}",
    "mapfc+ccbs": r"\textsc{MAPFC}+\textsf{CCBS}",
    "mapfc+ccbs+jl": r"\textsc{MAPFC}+\textsf{CCBS}$\rightarrow$\textsf{JL}",
}


def _fmt_pct(x: float | None) -> str:
    return r"--" if x is None else f"{100 * x:.1f}"


def _fmt_ms(x: float | None) -> str:
    return r"--" if x is None else f"{x * 1e3:.1f}"


def _stats(rows: list[dict]) -> dict[str, float | None]:
    if not rows:
        return {"n": 0, "isr": None, "saving": None, "total": None, "solver": None, "build": None}
    n = len(rows)
    isr = sum(1 for r in rows if r["isr"]) / n
    savings = [float(r["saving_ratio"]) for r in rows if r["saving_ratio"] is not None]
    saving = sum(savings) / len(savings) if savings else None
    total = statistics.median(float(r["runtime_total_s"]) for r in rows)
    solver = statistics.median(float(r["runtime_solver_only_s"]) for r in rows)
    build = statistics.median(float(r["build_runtime_s"]) for r in rows)
    return {
        "n": n,
        "isr": isr,
        "saving": saving,
        "total": total,
        "solver": solver,
        "build": build,
    }


def _aggregate_table(df: pl.DataFrame) -> str:
    lines: list[str] = [
        r"% Q4 aggregate per-config metrics. \input{...} from experiments.tex.",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Configuration & $n$ & ISR (\%) & Mean save (\%) & "
            r"Total (ms) & Solver (ms) & Build (ms) \\"
        ),
        r"\midrule",
    ]
    seen = set(df["config"].unique())
    for name in CONFIG_ORDER:
        if name not in seen:
            continue
        sub_rows = [r for r in df.iter_rows(named=True) if r["config"] == name]
        s = _stats(sub_rows)
        lines.append(
            f"{CONFIG_DISPLAY.get(name, name)} & "
            f"{s['n']} & "
            f"{_fmt_pct(s['isr'])} & "
            f"{_fmt_pct(s['saving'])} & "
            f"{_fmt_ms(s['total'])} & "
            f"{_fmt_ms(s['solver'])} & "
            f"{_fmt_ms(s['build'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def _per_family_table(df: pl.DataFrame) -> str:
    families = sorted({str(d) for d in df["dataset"].unique()})
    config_names = [c for c in CONFIG_ORDER if c in df["config"].unique()]

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in df.iter_rows(named=True):
        grouped[(row["dataset"], row["config"])].append(row)

    lines: list[str] = [
        r"% Q4 per-family per-config metrics. \input{...} from experiments.tex.",
        r"\begin{tabular}{ll" + "rrr" * len(config_names) + "}",
        r"\toprule",
        r"& "
        + " & ".join(r"\multicolumn{3}{c}{" + CONFIG_DISPLAY.get(c, c) + r"}" for c in config_names)
        + r" \\",
    ]
    lines.append(
        r"Family & " + " & ".join([r"ISR & Save & Solver (ms)"] * len(config_names)) + r" \\"
    )
    lines.append(r"\midrule")
    for fam in families:
        row = [fam]
        for cname in config_names:
            s = _stats(grouped.get((fam, cname), []))
            row.append(_fmt_pct(s["isr"]))
            row.append(_fmt_pct(s["saving"]))
            row.append(_fmt_ms(s["solver"]))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=PARQUET)
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix for output files, e.g. '_30s' -> q4_aggregate_30s.tex",
    )
    args = parser.parse_args()

    df = pl.read_parquet(args.parquet)
    print(f"loaded {df.height} rows ({df['config'].n_unique()} configs) from {args.parquet}")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    agg = _aggregate_table(df)
    agg_path = TABLES_DIR / f"q4_aggregate{args.suffix}.tex"
    agg_path.write_text(agg, encoding="utf-8")
    print(f"wrote {agg_path}")

    per_fam = _per_family_table(df)
    fam_path = TABLES_DIR / f"q4_per_family{args.suffix}.tex"
    fam_path.write_text(per_fam, encoding="utf-8")
    print(f"wrote {fam_path}")

    print("\n## Aggregate preview")
    print(agg)


if __name__ == "__main__":
    main()
