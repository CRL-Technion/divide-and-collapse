"""Dump POGEMA-derived schedules for the FULL 3,296-instance suite.

This is the full-coverage counterpart to ``dump_pogema_schedules.py`` (which
samples ``--per-cell`` scenarios). Here we enumerate *every* MAPF instance in
the five families and emit its per-agent input schedule, producing the input
file the re-evaluation harness consumes as ``--scope full``:

    benchmarks/schedules/pogema_full.jsonl

Expected per-dataset instance counts (one instance per (scenario, num_agents)
grid value; see ``mapfc.io.judgelight_dataset``):

    01-random 768, 02-mazes 768, 03-warehouse 768, 04-movingai 512,
    05-puzzles 480  --  total 3,296.

Like the harness, this exporter is **resumable**: a sidecar manifest
``<out>.done.txt`` lists every fully-written instance_id, and a rerun skips
those. A kill mid-instance simply re-runs that one instance on resume (the
agent rows are keyed by ``agent_idx`` downstream and trajectories are
deterministic, so a duplicate is harmless; a final dedupe pass rewrites a
clean canonical JSONL when the run completes). Progress is printed and mirrored
to ``<out_stem>_progress.json`` for polling.

Run in ``.venv-judgelight`` (the pinned pogema + pogema-benchmark +
pogema-toolbox stack), e.g.::

    .venv-judgelight/Scripts/python.exe scripts/dump_pogema_full.py

Inputs are frozen (POGEMA scenarios on disk, seed 0, deterministic
BatchAStarAgent), so the output is bit-for-bit reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JUDGELIGHT = REPO_ROOT / "third_party" / "judgelight"
sys.path.insert(0, str(JUDGELIGHT / "pogema"))
sys.path.insert(0, str(JUDGELIGHT / "pogema-toolbox"))
sys.path.insert(0, str(JUDGELIGHT / "pogema-benchmark" / "algorithms"))

from pogema import BatchAStarAgent, GridConfig, pogema_v0  # noqa: E402
from trajectory_collector import TrajectoryCollectorWrapper  # noqa: E402

from mapfc.io.judgelight_dataset import DATASETS, Instance, load_dataset  # noqa: E402

DATASET_ROOT = JUDGELIGHT / "pogema-benchmark" / "algorithms" / "experiments"
DEFAULT_OUT = REPO_ROOT / "benchmarks" / "schedules" / "pogema_full.jsonl"

EXPECTED_COUNTS = {
    "01-random": 768,
    "02-mazes": 768,
    "03-warehouse": 768,
    "04-movingai": 512,
    "05-puzzles": 480,
}


# --------------------------------------------------------------------------- #
# Episode (mirrors dump_pogema_schedules.py / Tang et al.'s build_env_config)  #
# --------------------------------------------------------------------------- #
def _build_env_config(instance: Instance) -> GridConfig:
    return GridConfig(
        on_target="nothing",
        collision_system="soft",
        observation_type="MAPF",
        max_episode_steps=128,
        map="\n".join(instance.grid),
        map_name=instance.map_name,
        seed=instance.seed,
        num_agents=instance.num_agents,
        agents_xy=[list(xy) for xy in instance.agents_xy],
        targets_xy=[list(xy) for xy in instance.targets_xy],
    )


def _run_episode(instance: Instance) -> list[list[tuple[int, int]]]:
    env = pogema_v0(grid_config=_build_env_config(instance))
    env = TrajectoryCollectorWrapper(env)
    algo = BatchAStarAgent()
    algo.reset_states()
    obs, _ = env.reset(seed=instance.seed)
    while True:
        obs, _rew, terminated, truncated, _infos = env.step(algo.act(obs))
        if all(terminated) or all(truncated):
            break
    return env.get_trajectories()


# --------------------------------------------------------------------------- #
# Resume manifest + progress                                                   #
# --------------------------------------------------------------------------- #
def _manifest_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".done.txt")


def _load_done(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    return {ln.strip() for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()}


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


def _write_progress(path: Path, *, done: int, total: int, started: float, per_ds_done, per_ds_total,
                    current: str, failed: int, status: str) -> None:
    elapsed = time.time() - started
    rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else None
    _atomic_write_json(
        path,
        {
            "status": status,
            "updated_at": time.time(),
            "current_dataset": current,
            "completed_instances": done,
            "total_instances": total,
            "failed_instances": failed,
            "percent": round(100.0 * done / total, 2) if total else 100.0,
            "elapsed_s": round(elapsed, 1),
            "rate_inst_per_s": round(rate, 3),
            "eta_s": round(eta, 1) if eta is not None else None,
            "per_dataset": {ds: {"done": per_ds_done.get(ds, 0), "total": per_ds_total.get(ds, 0)}
                            for ds in per_ds_total},
        },
    )


def _dedupe_rewrite(out: Path) -> int:
    """Rewrite ``out`` keeping the last row per (dataset, scenario, num_agents, agent_idx)."""
    by_key: dict[tuple, str] = {}
    with out.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            by_key[(e["dataset"], e["scenario"], e["num_agents"], e["agent_idx"])] = line
    tmp = out.with_suffix(out.suffix + ".dedupe.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for line in by_key.values():
            fh.write(line + "\n")
    tmp.replace(out)
    return len(by_key)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=sorted(DATASETS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None, help="Cap total instances (smoke test).")
    args = parser.parse_args()

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_path(out)
    progress_path = out.with_name(f"{out.stem}_progress.json")

    # If the JSONL was deleted but the manifest survived, start fresh so we do
    # not skip instances whose rows no longer exist.
    if not out.exists() and manifest.exists():
        manifest.unlink()

    print("enumerating instances ...", flush=True)
    all_instances: list[Instance] = []
    per_ds_total: dict[str, int] = {}
    for ds in args.datasets:
        insts = list(load_dataset(DATASET_ROOT / ds))
        per_ds_total[ds] = len(insts)
        all_instances.extend(insts)
        exp = EXPECTED_COUNTS.get(ds)
        flag = "" if exp is None or exp == len(insts) else f"  (EXPECTED {exp}!)"
        print(f"  {ds}: {len(insts)} instances{flag}", flush=True)
    if args.limit is not None:
        all_instances = all_instances[: args.limit]
        # Coverage is checked against what is actually in scope, not the full
        # on-disk dataset (the enumerated counts above already verified those).
        per_ds_total = defaultdict(int)
        for inst in all_instances:
            per_ds_total[inst.dataset] += 1
    total = len(all_instances)
    print(f"  total: {total} instances", flush=True)

    done = _load_done(manifest)
    per_ds_done: dict[str, int] = defaultdict(int)
    for iid in done:
        per_ds_done[iid.split("/", 1)[0]] += 1
    if done:
        print(f"  resuming: {len(done)}/{total} already done", flush=True)

    started = time.time()
    failed: list[tuple[str, str]] = []
    written_rows = 0
    processed = len(done)

    with out.open("a", encoding="utf-8") as fh, manifest.open("a", encoding="utf-8") as mf:
        for inst in all_instances:
            iid = inst.instance_id
            if iid in done:
                continue
            try:
                trajectories = _run_episode(inst)
            except Exception as exc:  # pragma: no cover - upstream pogema failures
                failed.append((iid, repr(exc)))
                continue
            block = "".join(
                json.dumps(
                    {
                        "dataset": inst.dataset,
                        "scenario": inst.scenario_name,
                        "num_agents": inst.num_agents,
                        "agent_idx": i,
                        "k_i": max(len(traj) - 1, 0),
                        "trajectory": [list(xy) for xy in traj],
                    }
                )
                + "\n"
                for i, traj in enumerate(trajectories)
            )
            fh.write(block)
            fh.flush()
            mf.write(iid + "\n")
            mf.flush()
            done.add(iid)
            per_ds_done[inst.dataset] += 1
            written_rows += len(trajectories)
            processed += 1
            if processed % args.progress_every == 0:
                _write_progress(progress_path, done=processed, total=total, started=started,
                                per_ds_done=per_ds_done, per_ds_total=per_ds_total,
                                current=inst.dataset, failed=len(failed), status="running")
                print(
                    f"  [{processed}/{total}] {inst.dataset} "
                    f"{100.0 * processed / total:.1f}%  {time.time() - started:.0f}s elapsed",
                    flush=True,
                )

    _write_progress(progress_path, done=processed, total=total, started=started,
                    per_ds_done=per_ds_done, per_ds_total=per_ds_total,
                    current=None, failed=len(failed), status="done")

    print(f"\nwrote ~{written_rows} new agent-trajectories this run", flush=True)
    if failed:
        print(f"FAILED: {len(failed)} instances", flush=True)
        for iid, exc in failed[:10]:
            print(f"  {iid}: {exc}", flush=True)

    # Canonical clean output only once everything (modulo failures) is in.
    if not failed:
        kept = _dedupe_rewrite(out)
        print(f"deduped canonical JSONL: {kept} agent rows in {out}", flush=True)

    # Final coverage check.
    print("\ncoverage:", flush=True)
    ok = True
    for ds in args.datasets:
        d, t = per_ds_done.get(ds, 0), per_ds_total.get(ds, 0)
        mark = "ok" if d == t else "INCOMPLETE"
        if d != t:
            ok = False
        print(f"  {ds}: {d}/{t} {mark}", flush=True)
    print(f"  total: {processed}/{total} {'COMPLETE' if ok and not failed else 'INCOMPLETE'}",
          flush=True)


if __name__ == "__main__":
    main()
