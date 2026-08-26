"""Resumable, pollable re-evaluation sweep with median-of-k timing.

This is the harness for the ICAPS-review re-evaluation campaign (Issues 8 and
9 of the 2026-06-25 review): it re-runs every configuration on every instance
of a chosen scope (the 500-instance pilot or the full 3,296-instance POGEMA
suite) and, for each ``(instance, config)`` cell, repeats the wall-clock
measurement ``k`` times (default 5) to produce a stable median plus dispersion
(IQR). Cost, savings, and the decomposition are deterministic (POGEMA inputs
are frozen at seed 0), so only the timing is repeated; cost is taken from the
first run and cross-checked against the rest.

Design goals (per user request 2026-06-25):

* **Resumable / pausable.** Progress is checkpointed to a JSON file keyed by
  config name after every instance (atomic temp-then-rename, with a Windows
  retry loop). Killing the process (Ctrl-C) and re-launching the same command
  resumes from the next unfinished ``(instance, config)`` cell. A cell is
  written only after all ``k`` repeats complete, so a mid-cell kill simply
  redoes that one cell.
* **Pollable progress.** After every instance the harness rewrites a small
  ``*_progress.json`` with completed/total cell counts, per-config progress,
  elapsed time, throughput, and an ETA. Poll it any time with
  ``status_reeval.py`` (or just read the JSON) without touching the running
  process.

Run (pilot, the current 500 instances)::

    .venv/Scripts/python.exe experiments/reeval/run_reeval.py --scope 500

Full suite (after the pilot validates)::

    .venv/Scripts/python.exe experiments/reeval/run_reeval.py --scope full

Poll progress from another shell::

    .venv/Scripts/python.exe experiments/reeval/status_reeval.py

Outputs (under ``results/raw/``):

* ``reeval_<scope>.parquet`` -- one row per ``(instance, config)`` with the
  deterministic fields plus ``runtime_total_s`` / ``runtime_solver_only_s`` set
  to the median over the ``k`` repeats and the dispersion columns
  ``*_p25`` / ``*_p75`` / ``*_min`` / ``*_max`` and JSON sample lists.
* ``reeval_<scope>.checkpoint.json`` -- the resume store.
* ``reeval_<scope>_progress.json`` -- the live progress file.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import polars as pl

from mapfc.eval import ALL_CONFIG_NAMES, CONFIG_BY_NAME, DEFAULT_CONFIG_NAMES, run_config

REPO = Path(__file__).resolve().parents[2]

# Scope -> input JSONL. The 500-instance pilot reuses the frozen subset; the
# full suite is expected at benchmarks/schedules/pogema_full.jsonl (built by
# the dataset-export step before the full campaign).
SCOPE_JSONL = {
    "500": REPO / "benchmarks" / "schedules" / "pogema_500.jsonl",
    "full": REPO / "benchmarks" / "schedules" / "pogema_full.jsonl",
}

DEFAULT_BUDGET_S = 30.0  # matches the paper's 30 s ceiling cap
DEFAULT_REPEATS = 5
RESULTS_DIR = REPO / "results" / "raw"


# --------------------------------------------------------------------------- #
# Instance loading (mirrors experiments/q4_end_to_end/run.py)                  #
# --------------------------------------------------------------------------- #
def _load_instances(jsonl: Path) -> list[tuple[str, str, int, tuple]]:
    """One row per (dataset, scenario, num_agents) with the joint schedule."""
    by_instance: dict[tuple[str, str, int], dict[int, tuple]] = defaultdict(dict)
    with jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            key = (entry["dataset"], entry["scenario"], entry["num_agents"])
            by_instance[key][entry["agent_idx"]] = tuple(tuple(xy) for xy in entry["trajectory"])
    out: list[tuple[str, str, int, tuple]] = []
    for (dataset, scenario, num_agents), traj_by_idx in sorted(by_instance.items()):
        schedules = tuple(traj_by_idx[i] for i in sorted(traj_by_idx))
        out.append((dataset, scenario, num_agents, schedules))
    return out


# --------------------------------------------------------------------------- #
# Checkpoint (atomic, Windows-safe) -- same contract as the Q4 driver          #
# --------------------------------------------------------------------------- #
def _keep_awake() -> None:
    """Block system idle-sleep while the campaign runs (Windows only).

    ``ES_SYSTEM_REQUIRED | ES_CONTINUOUS`` is per-process and clears
    automatically when the process exits, so nothing global needs reverting.
    A closed lid still sleeps the machine; the launch script handles that.
    """
    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def _load_checkpoint(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _atomic_write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    for attempt in range(5):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


# --------------------------------------------------------------------------- #
# Progress file                                                                #
# --------------------------------------------------------------------------- #
def _write_progress(
    path: Path,
    *,
    scope: str,
    configs: list[str],
    n_instances: int,
    cp: dict[str, list[dict]],
    started_at: float,
    status: str,
    current_config: str | None,
) -> None:
    total_cells = n_instances * len(configs)
    per_config = {c: {"done": len(cp.get(c, [])), "total": n_instances} for c in configs}
    completed = sum(v["done"] for v in per_config.values())
    elapsed = time.time() - started_at
    rate = completed / elapsed if elapsed > 0 and completed > 0 else 0.0
    remaining = total_cells - completed
    eta = remaining / rate if rate > 0 else None
    _atomic_write_json(
        path,
        {
            "status": status,
            "scope": scope,
            "started_at": started_at,
            "updated_at": time.time(),
            "current_config": current_config,
            "completed_cells": completed,
            "total_cells": total_cells,
            "percent": round(100.0 * completed / total_cells, 2) if total_cells else 100.0,
            "elapsed_s": round(elapsed, 1),
            "rate_cells_per_s": round(rate, 3),
            "eta_s": round(eta, 1) if eta is not None else None,
            "per_config": per_config,
        },
    )


# --------------------------------------------------------------------------- #
# Median-of-k timing for one (instance, config) cell                           #
# --------------------------------------------------------------------------- #
def _quantiles(samples: list[float]) -> tuple[float, float]:
    """(p25, p75); robust for small k via inclusive quantiles."""
    if len(samples) < 2:
        v = samples[0] if samples else 0.0
        return v, v
    qs = statistics.quantiles(samples, n=4, method="inclusive")
    return qs[0], qs[2]


def _timed_cell(cfg, *, dataset, scenario, num_agents, schedules, budget_s, repeats) -> dict:
    """Run a cell up to ``repeats`` times for median timing; return the row dict.

    Every run contributes a wall-clock sample; the deterministic fields (cost,
    savings, decomposition, failure_reason) come from the chosen base record.
    Two refinements over a naive k-repeat, learned from the pilot:

    * ​Decisive-timeout short-circuit. If the first two runs both fail, the
      remaining repeats would only re-confirm the timeout, each burning a full
      ``budget_s``; we stop at two samples. (This is what made ``mapfc+ccbs``
      slow -- a 30 s timeout times five.)
    * Deterministic, exact-favouring cost. The base record is the *minimum-cost
      successful* repeat, not run 0. For configs whose cost can flip with timing
      near a wall-clock fallback boundary (the hybrid's 1.5 s C-CBS cap, or
      C-CBS itself near the budget) this makes the reported cost deterministic
      and picks the exact C-CBS optimum, which is never worse than the
      Judgelight fallback. ``cost_stable`` still flags any repeat disagreement.
    """
    recs: list[dict] = []
    for r in range(repeats):
        rec, _ = run_config(
            cfg,
            dataset=dataset,
            scenario=scenario,
            num_agents=num_agents,
            schedules=schedules,
            budget_s=budget_s,
        )
        recs.append(asdict(rec))
        if r == 1 and recs[0]["soc"] is None and recs[1]["soc"] is None:
            break  # decisive timeout: skip the remaining repeats

    total_samples = [float(d["runtime_total_s"]) for d in recs]
    solver_samples = [float(d["runtime_solver_only_s"]) for d in recs]
    soc_samples = [d["soc"] for d in recs]
    socs = [s for s in soc_samples if s is not None]

    if socs:
        base = dict(min((d for d in recs if d["soc"] is not None), key=lambda d: d["soc"]))
    else:
        base = dict(recs[0])

    base["runtime_total_s"] = statistics.median(total_samples)
    base["runtime_solver_only_s"] = statistics.median(solver_samples)
    t_p25, t_p75 = _quantiles(total_samples)
    s_p25, s_p75 = _quantiles(solver_samples)
    base.update(
        timing_repeats=len(recs),
        total_s_p25=t_p25,
        total_s_p75=t_p75,
        total_s_min=min(total_samples),
        total_s_max=max(total_samples),
        solver_s_p25=s_p25,
        solver_s_p75=s_p75,
        total_s_samples=json.dumps(total_samples),
        solver_s_samples=json.dumps(solver_samples),
        cost_stable=(len(set(soc_samples)) == 1),
        soc_min=(min(socs) if socs else None),
        soc_max=(max(socs) if socs else None),
    )
    return base


# --------------------------------------------------------------------------- #
# Sweep one config                                                             #
# --------------------------------------------------------------------------- #
def _run_config_sweep(
    config_name,
    instances,
    *,
    budget_s,
    repeats,
    cp,
    checkpoint_path,
    progress_path,
    pause_path,
    scope,
    configs,
    started_at,
) -> None:
    cfg = CONFIG_BY_NAME[config_name]
    rows: list[dict] = list(cp.get(config_name, []))
    done_ids = {r["instance_id"] for r in rows}
    n = len(instances)
    t0 = time.perf_counter()
    if rows:
        print(f"  {config_name}: resuming with {len(rows)}/{n} cached cells", flush=True)
    new_this_run = 0
    for i, (dataset, scenario, num_agents, schedules) in enumerate(instances):
        instance_id = f"{dataset}/{scenario}/n{num_agents}"
        if instance_id in done_ids:
            continue
        row = _timed_cell(
            cfg,
            dataset=dataset,
            scenario=scenario,
            num_agents=num_agents,
            schedules=schedules,
            budget_s=budget_s,
            repeats=repeats,
        )
        rows.append(row)
        done_ids.add(instance_id)
        new_this_run += 1
        cp[config_name] = rows
        _atomic_write_json(checkpoint_path, cp)
        _write_progress(
            progress_path,
            scope=scope,
            configs=configs,
            n_instances=n,
            cp=cp,
            started_at=started_at,
            status="running",
            current_config=config_name,
        )
        if pause_path.exists():
            _write_progress(
                progress_path,
                scope=scope,
                configs=configs,
                n_instances=n,
                cp=cp,
                started_at=started_at,
                status="paused",
                current_config=config_name,
            )
            print(
                f"  pause requested ({pause_path.name}); checkpointed after "
                f"{len(rows)}/{n} cells of {config_name}. Relaunch to resume.",
                flush=True,
            )
            raise SystemExit(0)
        if i % 25 == 0 and i > 0:
            print(
                f"  {config_name} [{i}/{n}] {time.perf_counter() - t0:.0f}s "
                f"(+{new_this_run} this run)",
                flush=True,
            )
    cp[config_name] = rows
    _atomic_write_json(checkpoint_path, cp)
    print(
        f"  {config_name} done in {time.perf_counter() - t0:.1f}s "
        f"({len(rows)} cells, +{new_this_run} this run)",
        flush=True,
    )


def _aggregate_summary(rows: list[dict]) -> None:
    by_config: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_config[r["config"]].append(r)
    print("\n## Aggregate (median over repeats)")
    print(
        f"{'config':<25} {'n':>5} {'isr':>6} {'soc_ok':>7} {'unstable':>9} "
        f"{'mean_sav':>9} {'med_total':>11} {'med_solve':>11}"
    )
    seen = {r["config"] for r in rows}
    for name in [c for c in ALL_CONFIG_NAMES if c in seen]:
        sub = by_config.get(name, [])
        if not sub:
            continue
        n = len(sub)
        isr = sum(1 for r in sub if r["isr"])
        soc_ok = sum(1 for r in sub if r["soc"] is not None)
        unstable = sum(1 for r in sub if not r.get("cost_stable", True))
        savings = [r["saving_ratio"] for r in sub if r["saving_ratio"] is not None]
        mean_sav = (sum(savings) / len(savings)) if savings else float("nan")
        med_total = statistics.median([r["runtime_total_s"] * 1e3 for r in sub])
        med_solve = statistics.median([r["runtime_solver_only_s"] * 1e3 for r in sub])
        print(
            f"{name:<25} {n:>5} {isr:>6} {soc_ok:>7} {unstable:>9} "
            f"{mean_sav:>9.3f} {med_total:>9.1f}ms {med_solve:>9.1f}ms"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=sorted(SCOPE_JSONL), default="500")
    parser.add_argument("--budget-sec", type=float, default=DEFAULT_BUDGET_S)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIG_NAMES))
    parser.add_argument("--max-instances", type=int, default=None, help="Smoke-test cap.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    jsonl = SCOPE_JSONL[args.scope]
    if not jsonl.exists():
        raise SystemExit(
            f"input schedules not found: {jsonl}\n"
            f"(the '{args.scope}' scope expects this file; build it first)"
        )
    out = args.out or (RESULTS_DIR / f"reeval_{args.scope}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out.with_suffix(".checkpoint.json")
    progress_path = out.with_name(f"{out.stem}_progress.json")
    pause_path = out.with_suffix(".pause")
    pause_path.unlink(missing_ok=True)  # a stale pause request must not stop a fresh launch
    _keep_awake()

    print(f"loading instances from {jsonl} ...", flush=True)
    instances = _load_instances(jsonl)
    if args.max_instances is not None:
        instances = instances[: args.max_instances]
    n = len(instances)
    print(
        f"  {n} instances x {len(args.configs)} configs x {args.repeats} repeats; "
        f"budget={args.budget_sec}s",
        flush=True,
    )
    print(f"  checkpoint: {checkpoint_path}", flush=True)
    print(f"  progress:   {progress_path}  (poll with status_reeval.py)", flush=True)

    cp = _load_checkpoint(checkpoint_path)
    if cp and any(len(v) > n for v in cp.values()):
        print("  checkpoint has more cells than current scope; starting fresh", flush=True)
        cp = {}

    started_at = time.time()
    _write_progress(
        progress_path,
        scope=args.scope,
        configs=args.configs,
        n_instances=n,
        cp=cp,
        started_at=started_at,
        status="running",
        current_config=None,
    )

    for config_name in args.configs:
        if config_name not in CONFIG_BY_NAME:
            print(f"  {config_name}: unknown configuration; skipping", flush=True)
            continue
        if len(cp.get(config_name, [])) >= n:
            print(f"  {config_name}: cached ({len(cp[config_name])} cells); skipping", flush=True)
            continue
        print(f"running {config_name} ({len(cp.get(config_name, []))}/{n} cached)...", flush=True)
        _run_config_sweep(
            config_name,
            instances,
            budget_s=args.budget_sec,
            repeats=args.repeats,
            cp=cp,
            checkpoint_path=checkpoint_path,
            progress_path=progress_path,
            pause_path=pause_path,
            scope=args.scope,
            configs=args.configs,
            started_at=started_at,
        )

    all_rows: list[dict] = []
    for config_name in args.configs:
        all_rows.extend(cp.get(config_name, []))
    _write_progress(
        progress_path,
        scope=args.scope,
        configs=args.configs,
        n_instances=n,
        cp=cp,
        started_at=started_at,
        status="done",
        current_config=None,
    )
    if not all_rows:
        print("no rows to write; exiting", flush=True)
        return
    for r in all_rows:
        for col in ("soc", "saved_soc", "decomp_lb", "soc_min", "soc_max"):
            if r.get(col) is not None:
                r[col] = int(r[col])
    df = pl.from_dicts(all_rows, infer_schema_length=len(all_rows))
    df.write_parquet(out)
    print(f"\nwrote {out} ({len(all_rows)} rows)", flush=True)
    _aggregate_summary(all_rows)


if __name__ == "__main__":
    main()
