"""Hand-crafted edge cases for mapfc.single_agent.constrained_dag.solve_constrained."""

from __future__ import annotations

from mapfc.single_agent.collapse_dag import baseline_cost
from mapfc.single_agent.constrained_dag import solve_constrained


def test_empty_constraints_matches_baseline_on_no_repeats() -> None:
    schedule = ["a", "b", "c", "d"]
    result = solve_constrained(schedule, ())
    assert result is not None
    cost, plan = result
    assert cost == 3
    assert plan == schedule


def test_empty_constraints_matches_optimum_with_collapses() -> None:
    schedule = ["a", "b", "c", "b", "d"]
    result = solve_constrained(schedule, ())
    assert result is not None
    cost, plan = result
    assert cost == 2
    assert plan == ["a", "b", "b", "b", "d"]


def test_empty_schedule() -> None:
    result = solve_constrained([], ())
    assert result == (0, [])


def test_single_vertex_schedule_unconstrained() -> None:
    result = solve_constrained(["a"], ())
    assert result == (0, ["a"])


def test_start_blocked_is_infeasible() -> None:
    schedule = ["a", "b", "c"]
    assert solve_constrained(schedule, [("a", 0)]) is None


def test_end_blocked_is_infeasible() -> None:
    schedule = ["a", "b", "c"]
    assert solve_constrained(schedule, [("c", 2)]) is None


def test_blocked_interior_node_forces_collapse() -> None:
    schedule = ["a", "b", "c", "b", "d"]
    result = solve_constrained(schedule, [("c", 2)])
    assert result is not None
    cost, plan = result
    assert cost == 2
    assert plan == ["a", "b", "b", "b", "d"]


def test_blocked_interior_with_no_collapse_alternative_infeasible() -> None:
    schedule = ["a", "b", "c", "d"]
    assert solve_constrained(schedule, [("c", 2)]) is None


def test_forbidden_anchor_cell_splits_window() -> None:
    schedule = ["a", "b", "c", "b", "d", "b", "e"]
    cost_unforbid = solve_constrained(schedule, ())
    assert cost_unforbid is not None
    assert cost_unforbid[0] == 2
    forbidden = [("b", 4)]
    result = solve_constrained(schedule, forbidden)
    assert result is not None
    cost, _ = result
    assert cost > cost_unforbid[0]


def test_monotonicity_under_added_constraint() -> None:
    schedule = ["a", "b", "c", "b", "d", "b", "e"]
    base = solve_constrained(schedule, ())
    assert base is not None
    stricter = solve_constrained(schedule, [("c", 2)])
    if stricter is not None:
        assert stricter[0] >= base[0]


def test_running_example_agent3_under_no_constraints() -> None:
    schedule = ["d"] * 5
    result = solve_constrained(schedule, ())
    assert result == (0, schedule)


def test_grid_cell_vertex_type() -> None:
    schedule: list[tuple[int, int]] = [(0, 0), (0, 1), (0, 0), (1, 0)]
    result = solve_constrained(schedule, ())
    assert result is not None
    cost, plan = result
    assert cost == 1
    assert plan == [(0, 0), (0, 0), (0, 0), (1, 0)]


def test_grid_cell_with_forbidden_anchor() -> None:
    schedule: list[tuple[int, int]] = [(0, 0), (0, 1), (0, 0), (1, 0)]
    result = solve_constrained(schedule, [((0, 0), 1)])
    assert result is not None
    cost, plan = result
    assert plan[1] != (0, 0)
    assert cost <= baseline_cost(schedule)


def test_constraint_outside_horizon_ignored() -> None:
    schedule = ["a", "b", "c", "b"]
    result = solve_constrained(schedule, [("b", 99), ("z", -3)])
    assert result is not None
    cost, _ = result
    assert cost == solve_constrained(schedule, ())[0]


def test_lemma1_counterexample_with_no_constraints() -> None:
    """Constrained sweep collapses to unconstrained on empty C_i. The Lem. 1
    counterexample must yield cost 1 here too."""
    schedule = ["a", "c", "a", "b", "a", "c", "b"]
    result = solve_constrained(schedule, ())
    assert result is not None
    cost, plan = result
    assert cost == 1
    assert plan == ["a", "a", "a", "b", "b", "b", "b"]
