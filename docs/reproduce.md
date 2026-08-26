# Reproducing the paper

Every number, table, and figure in the paper comes from the pipeline below.
The expected values are stated so you can diff your run against ours.

Hardware used for the published timings: Intel Core Ultra 7 265U (14 logical
cores, 3.4 GHz base), Windows 11, Python 3.12, Gurobi 13.0 for the ILP
baseline. Our framework is pure Python throughout; Judgelight delegates to a
compiled commercial solver, so it holds a per-component implementation
advantage that the reported speedups are measured *against*.

---

## 0. Setup

```bash
uv venv && uv pip install -e ".[dev,viz]"
pytest -m "not slow"                     # should be green before you start
```

For anything involving Judgelight (the baseline, and the hybrid's fallback):

```bash
bash scripts/setup_judgelight.sh         # or scripts/setup_judgelight.ps1
```

This clones Tang et al.'s solver at the SHAs pinned in
`third_party/SUBMODULE_SHAS.json`. It needs a working Gurobi installation and
license. See `../THIRD-PARTY.md`.

## 1. Input data

The input plans are POGEMA scenarios solved by POGEMA's `BatchAStarAgent`. We
ship none of them; regenerate with:

```bash
python scripts/dump_pogema_full.py       # -> benchmarks/schedules/pogema_full.jsonl
```

Deterministic (pinned SHAs, seed 0), resumable, and about 300 MB. Expect
3,296 instances: 768 each in 01-random, 02-mazes, and 03-warehouse, 512 in
04-movingai, 480 in 05-puzzles.

## 2. Q1 — the per-agent primitive

```bash
python experiments/q1_single_agent/bench_pogema.py
```

Runs the primitive in isolation on every agent schedule, on prefixes spanning
`k = 8..128` (about 1.5 million solves). **Expected: median 10 microseconds,
no solve above 150 microseconds.**

## 3. Q2 — how the interaction graph decomposes

```bash
python experiments/q2_decomposition/run.py
python experiments/q2_decomposition/figures.py     # -> Fig. 2
```

**Expected: 43.8% of agents in singleton components overall; median H-build
3.0 ms; per-family singleton shares 45.0 / 37.9 / 32.6 / 37.7 / 75.9%.**

## 4. Q3 — C-CBS against Judgelight, per component

```bash
python experiments/comparison/run_joint_comparison.py --max-size 50
python experiments/comparison/run_joint_comparison.py --min-size 51 --max-size 224
python experiments/comparison/run_ablation.py --max-size 50
python experiments/comparison/plot_ablation.py     # -> Fig. 3
```

The two comparison runs produce the 7,322 components of size at most 50 and
the 969-component tail; together they are Tbl. 2. The ablation is the
four-rung ladder of App. B.1 over the same 7,322 components.

**Expected: C-CBS wins every size bucket up to 29 agents; Judgelight takes the
lead past the [30, 39] band; on the tail C-CBS solves 38.2% within 5 s against
Judgelight's 100%, which is what motivates the size gate at tau = 40.**

## 5. Q4 — the end-to-end campaign

This is the expensive one: seven configurations over all 3,296 instances, five
timing repeats each, a 30-second per-instance limit. **Budget about a day of
wall-clock.**

```bash
python experiments/reeval/run_reeval.py --scope full --repeats 5 \
    --configs indep_lb nocollapse mapfc+ccbs+jl+gate40 judgelight \
              mapfc+judgelight mapfc+ccbs+jl mapfc+ccbs \
    --out results/raw/reeval_arxiv.parquet
```

The harness checkpoints after every `(instance, configuration)` cell, so it is
safe to interrupt: relaunch the same command and it resumes. Poll progress
from another shell with `python experiments/reeval/status_reeval.py`. A cell
whose first two runs both fail is recorded as unsolved without the remaining
three repeats.

Smoke test first if you like:

```bash
python experiments/reeval/run_reeval.py --scope full --repeats 1 \
    --max-instances 20 --configs indep_lb mapfc+ccbs \
    --out results/raw/smoke.parquet
```

## 6. Tables and figures

```bash
python experiments/reeval/paper_numbers_q4.py      # Tbl. 3, Tbl. 4, Q4 prose
python experiments/reeval/paper_numbers_full.py    # Q1/Q2/Q3 prose numbers
python experiments/q4_end_to_end/plot_cactus.py    # -> Fig. 4
```

`paper_numbers_q4.py` reads `results/raw/reeval_arxiv.parquet` and prints
every Q4 and App. B.2 quantity the paper states. Expected headline values:

| Quantity | Expected |
|---|---|
| `DnC+Hyb` median wall-clock | 43.4 ms |
| `Judgelight` median wall-clock | 420.9 ms |
| Ratio of medians | 9.7x |
| Median per-instance speedup | 10.5x |
| `DnC+Hyb` success rate | 3,296 / 3,296 |
| `DnC+C-CBS` success rate | 2,815 / 3,296 (85.4%) |
| Cost: hybrid ties / wins / loses vs Judgelight | 2,697 / 599 / 0 |
| Mean saving, hybrid vs Judgelight | 0.360 vs 0.359 |
| Coordination-free instances | 437, where the framework is about 1,900x faster |

Timings will differ on your hardware; the ratios and every cost-based number
should not, since costs are deterministic.

## Notes on determinism

- Costs, savings, and the decomposition are deterministic: POGEMA inputs are
  frozen at seed 0 and the solvers are exact. Only wall-clock varies.
- For budget-limited configurations the *returned plan* can differ between
  repeats near the time limit; the harness reports the cheapest successful run
  and flags any disagreement in `cost_stable`.
- Run on AC power with a quiet machine. On a laptop, battery power throttles
  the CPU and inflates every timing.
