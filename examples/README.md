# Examples

Three small, self-contained runs. Everything here is first-party: the input
instances are written by `data/make_examples.py`, not taken from any external
benchmark.

## `running_example.py`

The paper's Fig. 1 worked end to end: three agents on a six-vertex star, where
agents 1 and 2 both pass through the hub `b` and agent 3 sits alone on a leaf.
Prints the input plan, the interaction graph, the decomposition, and the
optimum.

```
python examples/running_example.py
```

Expected: input cost 8, one non-trivial component `{0, 1}`, one singleton, and
an optimum of cost 4 (saving ratio 0.5).

## `solve_instance.py`

A small command-line driver: give it a JSONL file of instances and it reports,
per instance, the decomposition and the collapsed cost.

```
python examples/solve_instance.py examples/data/toy_contention.jsonl
python examples/solve_instance.py examples/data/toy_mixed.jsonl --config indep_lb
python examples/solve_instance.py --list-configs
```

The default configuration `mapfc+ccbs` is the exact solver and needs nothing
beyond this package. Configurations whose names mention Judgelight additionally
require Tang et al.'s solver and Gurobi (see `../THIRD-PARTY.md`).

## `data/`

| File | What it shows |
|---|---|
| `toy_singletons.jsonl` | Two agents with disjoint reach sets: no interaction edges, both dispatched as singletons, cost 8 → 4. |
| `toy_contention.jsonl` | The paper's running example in grid coordinates: both agents want the hub, so they form one non-trivial component the joint solver must arbitrate. Cost 8 → 4. |
| `toy_mixed.jsonl` | Six agents: one contending pair plus four independents, exercising singleton dispatch and the joint solver in the same instance. Cost 24 → 12. |

Regenerate them at any time:

```
python examples/data/make_examples.py
```
