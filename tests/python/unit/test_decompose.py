"""Hand-crafted edge cases for mapfc.decompose."""

from __future__ import annotations

from mapfc.decompose import (
    DecompositionResult,
    connected_components,
    decompose,
    edge_witnesses,
    interaction_graph,
    reach_set,
)


def test_reach_single_vertex() -> None:
    assert reach_set(["a"]) == {("a", 0)}


def test_reach_no_repeats() -> None:
    cells = reach_set(["a", "b", "c", "d"])
    assert cells == {("a", 0), ("b", 1), ("c", 2), ("d", 3)}


def test_reach_single_anchor() -> None:
    cells = reach_set(["a", "b", "c", "b", "d"])
    assert cells == {
        ("a", 0),
        ("b", 1),
        ("c", 2),
        ("b", 3),
        ("d", 4),
        ("b", 2),
    }


def test_reach_anchor_at_endpoints() -> None:
    cells = reach_set(["a", "b", "c", "a"])
    assert cells == {("a", 0), ("a", 1), ("a", 2), ("a", 3), ("b", 1), ("c", 2)}


def test_reach_all_same() -> None:
    cells = reach_set(["d", "d", "d", "d"])
    assert cells == {("d", 0), ("d", 1), ("d", 2), ("d", 3)}


def test_reach_empty() -> None:
    assert reach_set([]) == set()


def test_interaction_disjoint_reaches() -> None:
    schedules = [["a", "b"], ["c", "d"]]
    n, edges = interaction_graph(schedules)
    assert n == 2
    assert edges == set()


def test_interaction_single_shared_cell() -> None:
    schedules = [["a", "b"], ["c", "b"]]
    n, edges = interaction_graph(schedules)
    assert n == 2
    assert edges == {frozenset({0, 1})}


def test_interaction_three_way_shared_cell() -> None:
    schedules = [["a", "x"], ["b", "x"], ["c", "x"]]
    n, edges = interaction_graph(schedules)
    assert n == 3
    assert edges == {frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2})}


def test_interaction_anchor_extends_reach() -> None:
    schedules = [["a", "b", "c", "b", "e"], ["x", "x", "b", "x", "x"]]
    n, edges = interaction_graph(schedules)
    assert n == 2
    assert edges == {frozenset({0, 1})}
    witnesses = edge_witnesses(schedules)
    cell = witnesses[frozenset({0, 1})]
    assert cell == ("b", 2)


def test_components_empty_graph() -> None:
    components, component_of = connected_components(3, [])
    assert components == [frozenset({0}), frozenset({1}), frozenset({2})]
    assert component_of == (0, 1, 2)


def test_components_chain() -> None:
    components, component_of = connected_components(4, [frozenset({0, 1}), frozenset({1, 2})])
    assert components == [frozenset({3}), frozenset({0, 1, 2})]
    assert component_of == (1, 1, 1, 0)


def test_components_two_islands() -> None:
    components, _ = connected_components(
        5, [frozenset({0, 1}), frozenset({2, 3}), frozenset({3, 4})]
    )
    sizes = sorted(len(c) for c in components)
    assert sizes == [2, 3]


def test_decompose_running_example() -> None:
    schedules = [
        ["a", "b", "c", "b", "e"],
        ["b", "a", "b", "a", "b"],
        ["d", "d", "d", "d", "d"],
    ]
    result = decompose(schedules)
    assert isinstance(result, DecompositionResult)
    assert result.n_agents == 3
    assert result.singletons == (2,)
    assert result.non_trivial_components == (frozenset({0, 1}),)
    assert result.n_singletons == 1
    assert result.singleton_fraction == 1 / 3
    assert result.max_non_trivial_size == 2
    assert result.component_size_histogram() == {1: 1, 2: 1}
    assert result.edges_count == 1


def test_decompose_all_singletons() -> None:
    schedules = [["a"], ["b"], ["c"]]
    result = decompose(schedules)
    assert result.n_singletons == 3
    assert result.non_trivial_components == ()
    assert result.singleton_fraction == 1.0
    assert result.edges_count == 0


def test_decompose_one_big_component() -> None:
    schedules = [["x", "a"], ["x", "b"], ["x", "c"]]
    result = decompose(schedules)
    assert result.n_singletons == 0
    assert result.non_trivial_components == (frozenset({0, 1, 2}),)
    assert result.singleton_fraction == 0.0
    assert result.max_non_trivial_size == 3


def test_decompose_empty_schedules_list() -> None:
    result = decompose([])
    assert result.n_agents == 0
    assert result.singletons == ()
    assert result.non_trivial_components == ()
    assert result.singleton_fraction == 0.0
    assert result.edges_count == 0
