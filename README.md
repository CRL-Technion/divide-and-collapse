# Divide and Collapse

Exact decomposition of **MAPF-Collapse** into independent sub-instances.

Reference implementation and full experimental pipeline for *Divide and
Collapse: MAPF-Collapse via Exact Decomposition into Independent Sub-Instances*
(Oren Salzman, Technion).

## The problem

A MAPF solver hands you a feasible plan. Many of the moves in it are pointless:
an agent wanders away from a cell and comes back, burning battery without
changing what the plan achieves. **MAPF-Collapse**, formalized by Tang et al.
(WAFR 2026), asks for the cheapest plan obtainable by *collapsing* such closed
subwalks into waits, while keeping the plan collision-free and the timeline
intact. It is NP-hard, and the prior solver compiles the whole instance into a
single ILP.

## The idea

Most agents never contend with anyone. If two agents can never occupy the same
cell at the same time under *any* sequence of collapses, nothing either does
can affect the other, so they can be optimized separately. This repository
computes that split exactly and up front:

1. **Reach sets.** For each agent, the set of space-time cells it could ever
   occupy under any sequence of collapses: its input trajectory plus a
   `[first visit, last visit]` band at every revisited vertex.
2. **Interaction graph.** Join two agents when their reach sets intersect. The
   connected components are provably independent sub-instances, so solving them
   separately loses no optimality.
3. **Dispatch.** Singleton components — the large majority — reduce to a
   shortest path in a *Collapse DAG* and are solved in time linear in the plan
   length. Only the small coupled residue goes to a joint solver: either
   `C-CBS`, an exact CBS adaptation built on a linear-time constrained sweep,
   or Judgelight, or a regime-aware hybrid of the two.

On the 3,296-instance POGEMA suite the recommended hybrid solves every
instance, never returns a plan costlier than Judgelight, is strictly cheaper on
18% of them, and is about 10x faster in median wall-clock.

## Install

Pure Python, no compiler, no commercial solver needed for our own solvers.

```bash
git clone <this repo> && cd mapf-compress
uv venv && uv pip install -e ".[dev,viz]"    # or: pip install -e ".[dev,viz]"
```

Check it works:

```bash
python examples/running_example.py
pytest -m "not slow"
```

## Quick start

```bash
# the paper's Fig. 1, decomposed and solved
python examples/running_example.py

# your own instances, in the harness's JSONL format
python examples/solve_instance.py examples/data/toy_mixed.jsonl
python examples/solve_instance.py --list-configs
```

See [`examples/README.md`](examples/README.md) for the input format and what
each bundled instance demonstrates.

## Layout

```
src/python/mapfc/
  decompose/      reach sets, interaction graph, connected components
  single_agent/   Collapse DAG + the linear-time sweep (constrained and not)
  joint/          C-CBS, and the Judgelight wrapper
  eval/           the configurations evaluated in the paper, and the driver
  io/             POGEMA/Judgelight dataset readers
experiments/      the Q1-Q4 drivers and the re-evaluation harness
examples/         small self-contained runs (start here)
scripts/          data regeneration and Judgelight setup
tests/            unit, property, regression, and stress tests
```

## Reproducing the paper

Full instructions, with expected numbers, are in
[`docs/reproduce.md`](docs/reproduce.md). In short:

```bash
bash scripts/setup_judgelight.sh              # baseline (see THIRD-PARTY.md)
python scripts/dump_pogema_full.py            # regenerate the input plans
python experiments/reeval/run_reeval.py --scope full --repeats 5 \
    --configs indep_lb nocollapse mapfc+ccbs+jl+gate40 judgelight \
              mapfc+judgelight mapfc+ccbs+jl mapfc+ccbs
python experiments/reeval/paper_numbers_q4.py # every Q4 number
```

The full campaign takes roughly a day of compute; add `--max-instances 20` for
a smoke run that finishes in minutes.

## Data and third-party code

This repository ships no third-party code and no third-party data. The
Judgelight baseline is fetched from its authors at pinned commits, and the
POGEMA-derived input plans are regenerated locally and deterministically.
Details and licensing in [`THIRD-PARTY.md`](THIRD-PARTY.md).

## Citing

See [`CITATION.cff`](CITATION.cff), or:

```bibtex
@misc{Salzman2026DivideAndCollapse,
  author = {Oren Salzman},
  title  = {Divide and Collapse: {MAPF-Collapse} via Exact Decomposition
            into Independent Sub-Instances},
  year   = {2026}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
