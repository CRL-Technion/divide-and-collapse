"""The paper's running example, end to end.

Three agents on the six-vertex star of Fig. 1. Agents 1 and 2 both pass
through the hub ``b``, so they interact and must be solved jointly; agent 3
sits on its own leaf and never interacts, so it is a singleton and is solved
in linear time by the per-agent primitive.

Run it after ``pip install -e .``::

    python examples/running_example.py

Expected output: an input cost of 8, an optimum of 4, one non-trivial
component ``{0, 1}`` and one singleton, and a saving ratio of 0.5.
"""

from __future__ import annotations

from mapfc.decompose import decompose, interaction_graph
from mapfc.joint import solve_ccbs
from mapfc.single_agent.collapse_dag import baseline_cost, solve_unconstrained

# Fig. 1(b): one row per agent, one column per timestep. Agent 1 walks
# a -> b -> c -> b -> e; agent 2 oscillates b -> f -> b -> f -> b; agent 3
# waits at d throughout.
SCHEDULES = (
    ("a", "b", "c", "b", "e"),
    ("b", "f", "b", "f", "b"),
    ("d", "d", "d", "d", "d"),
)
LABELS = ("agent 1", "agent 2", "agent 3")


def show(title: str, plans) -> None:
    print(f"\n{title}")
    for label, plan in zip(LABELS, plans, strict=True):
        print(f"  {label}: {' '.join(plan)}   cost {baseline_cost(plan)}")


def main() -> None:
    show("Input plan M", SCHEDULES)
    print(f"\n  total cost: {sum(baseline_cost(s) for s in SCHEDULES)}")

    # --- Step 1: decompose -------------------------------------------------
    decomp = decompose(list(SCHEDULES))
    _, edges = interaction_graph(list(SCHEDULES))
    print("\nInteraction graph H")
    print(f"  edges: {sorted(tuple(sorted(e)) for e in edges)}")
    print(f"  singletons: {list(decomp.singletons)}")
    print(f"  non-trivial components: {[sorted(c) for c in decomp.non_trivial_components]}")
    print(f"  singleton fraction: {decomp.singleton_fraction:.2f}")

    # --- Step 2: solve each component independently ------------------------
    plans: list[tuple[str, ...]] = [()] * len(SCHEDULES)

    for i in decomp.singletons:
        _, plan = solve_unconstrained(SCHEDULES[i])
        plans[i] = tuple(plan)

    for comp in decomp.non_trivial_components:
        members = sorted(comp)
        joint = solve_ccbs([SCHEDULES[i] for i in members])
        assert joint is not None, "C-CBS failed on the running example"
        for slot, i in enumerate(members):
            plans[i] = tuple(joint.plans[slot])

    # --- Step 3: report ----------------------------------------------------
    show("Optimal plan", plans)
    before = sum(baseline_cost(s) for s in SCHEDULES)
    after = sum(baseline_cost(p) for p in plans)
    print(f"\n  total cost: {after}   (was {before})")
    print(f"  saving ratio: {1 - after / before:.3f}")


if __name__ == "__main__":
    main()
