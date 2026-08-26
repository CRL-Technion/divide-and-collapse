"""End-to-end Q4 sweep: every configuration on every POGEMA instance.

For each instance in ``benchmarks/schedules/pogema_500.jsonl`` and each
configuration in ``mapfc.eval.ALL_CONFIG_NAMES``, runs the end-to-end
driver under a common per-instance budget (default 5 s, matching the
paper's specification) and writes one parquet row per
``(instance, config)`` pair to ``results/raw/q4_end_to_end.parquet``.

Resumable via a JSON checkpoint keyed by config name: a config that
already has ``n_instances`` cached rows is skipped on rerun. Instance
ordering is deterministic — sort by ``(dataset, scenario, num_agents)``
— so positional indexing is stable.

Stdout summary at the end: per-config success rate, mean saving ratio,
median runtime split.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import polars as pl

from mapfc.eval import ALL_CONFIG_NAMES, CONFIG_BY_NAME, DEFAULT_CONFIG_NAMES, run_config

REPO = Path(__file__).resolve().parents[2]
JSONL_IN = REPO / "benchmarks" / "schedules" / "pogema_500.jsonl"
PARQUET_OUT = REPO / "results" / "raw" / "q4_end_to_end.parquet"

DEFAULT_BUDGET_S = 5.0


def _load_instances() -> list[tuple[str, str, int, tuple[tuple[tuple[int, int], ...], ...]]]:
    """Return one row per (dataset, scenario, num_agents) with the joint schedule.

    The on-disk JSONL has one row per (instance, agent); we group by
    instance and rebuild the ordered list of per-agent trajectories.
    """
    by_instance: dict[tuple[str, str, int], dict[int, tuple[tuple[int, int], ...]]] = defaultdict(
        dict
    )
    with JSONL_IN.open(encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            key = (entry["dataset"], entry["scenario"], entry["num_agents"])
            by_instance[key][entry["agent_idx"]] = tuple(tuple(xy) for xy in entry["trajectory"])
    out: list[tuple[str, str, int, tuple[tuple[tuple[int, int], ...], ...]]] = []
    for (dataset, scenario, num_agents), traj_by_idx in sorted(by_instance.items()):
        schedules = tuple(traj_by_idx[i] for i in sorted(traj_by_idx))
        out.append((dataset, scenario, num_agents, schedules))
    return out


def _checkpoint_path(out: Path) -> Path:
    return out.with_suffix(".checkpoint.json")


def _load_checkpoint(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_checkpoint(path: Path, cp: dict[str, list[dict]]) -> None:
    """Atomic write: dump to .tmp then rename, so a kill mid-write cannot
    leave a half-written JSON that the next resume would fail to parse.

    Windows occasionally returns ``PermissionError`` on the rename when
    Defender briefly holds the freshly-closed file open for scanning;
    a short retry loop covers that transient.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cp), encoding="utf-8")
    for attempt in range(5):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def _run_config_sweep(
    config_name: str,
    instances: list,
    budget_s: float,
    cp: dict[str, list[dict]],
    checkpoint_path: Path,
    save_every: int = 1,
) -> list[dict]:
    """Run one configuration across every instance, returning per-row dicts.

    Resumable at per-instance granularity: rows already present in
    ``cp[config_name]`` (matched by instance_id) are skipped. The
    checkpoint is rewritten after every ``save_every`` instances so a
    crash loses at most that many rows of work.
    """
    cfg = CONFIG_BY_NAME[config_name]
    rows: list[dict] = list(cp.get(config_name, []))
    done_ids: set[str] = {r["instance_id"] for r in rows}
    t0 = time.perf_counter()
    n = len(instances)
    if rows:
        print(
            f"  {config_name}: resuming with {len(rows)}/{n} cached rows",
            flush=True,
        )
    new_this_run = 0
    for i, (dataset, scenario, num_agents, schedules) in enumerate(instances):
        instance_id = f"{dataset}/{scenario}/n{num_agents}"
        if instance_id in done_ids:
            continue
        rec, _ = run_config(
            cfg,
            dataset=dataset,
            scenario=scenario,
            num_agents=num_agents,
            schedules=schedules,
            budget_s=budget_s,
        )
        rows.append(asdict(rec))
        done_ids.add(instance_id)
        new_this_run += 1
        if new_this_run % save_every == 0 or i == n - 1:
            cp[config_name] = rows
            _save_checkpoint(checkpoint_path, cp)
        if i % 50 == 0 and i > 0:
            print(
                f"  {config_name} [{i}/{n}] {time.perf_counter() - t0:.0f}s "
                f"(new this run: {new_this_run})",
                flush=True,
            )
    cp[config_name] = rows
    _save_checkpoint(checkpoint_path, cp)
    print(
        f"  {config_name} done in {time.perf_counter() - t0:.1f}s "
        f"({len(rows)} rows, +{new_this_run} this run)",
        flush=True,
    )
    return rows


def _aggregate_summary(rows: list[dict]) -> None:
    """Stdout: per-config success rate, mean saving ratio, median runtime split."""
    by_config: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_config[r["config"]].append(r)
    print("\n## Aggregate")
    print(
        f"{'config':<25} {'n':>4} {'isr':>6} {'soc_ok':>7} "
        f"{'mean_sav':>9} {'med_total':>10} {'med_solve':>10}"
    )
    seen_configs = {r["config"] for r in rows}
    config_order = [n for n in ALL_CONFIG_NAMES if n in seen_configs]
    for name in config_order:
        sub = by_config.get(name, [])
        if not sub:
            continue
        n = len(sub)
        isr = sum(1 for r in sub if r["isr"])
        soc_ok = sum(1 for r in sub if r["soc"] is not None)
        savings = [r["saving_ratio"] for r in sub if r["saving_ratio"] is not None]
        mean_sav = (sum(savings) / len(savings)) if savings else float("nan")
        total_ms = [r["runtime_total_s"] * 1e3 for r in sub]
        solve_ms = [r["runtime_solver_only_s"] * 1e3 for r in sub]
        med_total = statistics.median(total_ms)
        med_solve = statistics.median(solve_ms)
        print(
            f"{name:<25} {n:>4} {isr:>6} {soc_ok:>7} "
            f"{mean_sav:>9.3f} {med_total:>8.1f}ms {med_solve:>8.1f}ms"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PARQUET_OUT)
    parser.add_argument("--budget-sec", type=float, default=DEFAULT_BUDGET_S)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIG_NAMES))
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Cap the number of instances loaded — useful for quick smoke runs.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Rewrite the checkpoint after this many newly-solved instances (default: 1).",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(args.out)

    print(f"loading instances from {JSONL_IN}...", flush=True)
    instances = _load_instances()
    if args.max_instances is not None:
        instances = instances[: args.max_instances]
    print(f"  {len(instances)} instances in scope", flush=True)

    cp = _load_checkpoint(checkpoint_path)
    # Per-instance resumption: we no longer require a config's cached row
    # list to match the current instance count exactly. A config with
    # 250/500 rows simply resumes from instance 250 inside
    # _run_config_sweep. We only drop the cache entirely if a config's
    # cached row count *exceeds* the current scope (e.g., switching to a
    # smaller --max-instances).
    if cp and any(len(v) > len(instances) for v in cp.values()):
        print("  checkpoint has more rows than current scope; starting fresh", flush=True)
        cp = {}

    for config_name in args.configs:
        if config_name not in CONFIG_BY_NAME:
            print(f"  {config_name}: unknown configuration; skipping", flush=True)
            continue
        cached = len(cp.get(config_name, []))
        if cached >= len(instances):
            print(f"  {config_name}: cached ({cached} rows); skipping", flush=True)
            continue
        print(
            f"running config {config_name} (budget={args.budget_sec}s, "
            f"{cached}/{len(instances)} cached)...",
            flush=True,
        )
        _run_config_sweep(
            config_name,
            instances,
            args.budget_sec,
            cp,
            checkpoint_path,
            save_every=args.save_every,
        )

    all_rows: list[dict] = []
    for config_name in args.configs:
        all_rows.extend(cp.get(config_name, []))

    if not all_rows:
        print("no rows to write; exiting", flush=True)
        return

    # Coerce nullable int columns to consistent Python types so polars'
    # schema inference doesn't crash on mixed-row int / None columns.
    for r in all_rows:
        for col in ("soc", "saved_soc", "decomp_lb"):
            if r.get(col) is not None:
                r[col] = int(r[col])
    df = pl.from_dicts(all_rows, infer_schema_length=len(all_rows))
    df.write_parquet(args.out)
    print(f"wrote {args.out} ({len(all_rows)} rows)", flush=True)

    _aggregate_summary(all_rows)


if __name__ == "__main__":
    main()
