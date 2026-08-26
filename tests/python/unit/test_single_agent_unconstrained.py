"""Hand-crafted edge cases for mapfc.single_agent.collapse_dag.solve_unconstrained."""

from __future__ import annotations

import pytest

from mapfc.single_agent.collapse_dag import (
    baseline_cost,
    maximal_loops,
    solve_unconstrained,
)


def test_one_vertex_schedule() -> None:
    cost, plan = solve_unconstrained(["a"])
    assert cost == 0
    assert plan == ["a"]


def test_all_wait() -> None:
    schedule = ["a", "a", "a", "a"]
    assert baseline_cost(schedule) == 0
    cost, plan = solve_unconstrained(schedule)
    assert cost == 0
    assert plan == schedule


def test_all_move_no_repeats() -> None:
    schedule = ["a", "b", "c", "d"]
    assert baseline_cost(schedule) == 3
    assert maximal_loops(schedule) == []
    cost, plan = solve_unconstrained(schedule)
    assert cost == 3
    assert plan == schedule


def test_single_anchor_visited_twice() -> None:
    schedule = ["a", "b", "c", "b", "d"]
    loops = maximal_loops(schedule)
    assert loops == [(1, 3)]
    assert baseline_cost(schedule) == 4
    cost, plan = solve_unconstrained(schedule)
    assert cost == 2
    assert plan == ["a", "b", "b", "b", "d"]


def test_nested_loops_outer_wins() -> None:
    schedule = ["a", "b", "c", "b", "c", "b", "d"]
    loops = maximal_loops(schedule)
    assert loops == [(1, 5), (2, 4)]
    assert baseline_cost(schedule) == 6
    cost, plan = solve_unconstrained(schedule)
    assert cost == 2
    assert plan == ["a", "b", "b", "b", "b", "b", "d"]


def test_anchor_at_start_and_end() -> None:
    schedule = ["a", "b", "c", "a"]
    loops = maximal_loops(schedule)
    assert loops == [(0, 3)]
    assert baseline_cost(schedule) == 3
    cost, plan = solve_unconstrained(schedule)
    assert cost == 0
    assert plan == ["a", "a", "a", "a"]


def test_two_disjoint_loops() -> None:
    schedule = ["a", "b", "a", "c", "d", "c", "e"]
    loops = maximal_loops(schedule)
    assert loops == [(0, 2), (3, 5)]
    assert baseline_cost(schedule) == 6
    cost, plan = solve_unconstrained(schedule)
    assert cost == 2
    assert plan == ["a", "a", "a", "c", "c", "c", "e"]


def test_grid_cell_vertex_type() -> None:
    schedule: list[tuple[int, int]] = [(0, 0), (0, 1), (0, 0), (1, 0)]
    cost, plan = solve_unconstrained(schedule)
    assert cost == 1
    assert plan == [(0, 0), (0, 0), (0, 0), (1, 0)]


def test_cost_never_exceeds_baseline() -> None:
    schedules = [
        ["a"],
        ["a", "b"],
        ["a", "a"],
        ["a", "b", "a"],
        ["a", "b", "c", "d", "e"],
        ["a", "b", "c", "b", "a"],
    ]
    for sched in schedules:
        cost, _ = solve_unconstrained(sched)
        assert cost <= baseline_cost(sched), sched


@pytest.mark.parametrize(
    "sched,expected_cost",
    [
        ([], 0),
        (["a"], 0),
        (["a", "a"], 0),
        (["a", "b"], 1),
        (["a", "b", "a"], 0),
    ],
)
def test_micro(sched: list[str], expected_cost: int) -> None:
    cost, _ = solve_unconstrained(sched)
    assert cost == expected_cost
