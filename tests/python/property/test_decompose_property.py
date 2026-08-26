"""Property tests for mapfc.decompose.

Invariants:

- Reach: ``Reach^i_orig subset Reach^i``; for every multi-visit anchor ``v``,
  every ``t in [first(v), last(v)]`` is in ``Reach^i``.
- Interaction graph: every edge ``{i, j}`` has a witness cell in
  ``Reach^i intersect Reach^j``; cross-component agent pairs have disjoint
  reach sets.
- Components: partition of ``{0, ..., n-1}``; agent-count conservation
  ``sum(|c|) == n``; ``component_of[a]`` indexes a set containing ``a``.
- Pipeline: singletons disjoint from non-trivial component members.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mapfc.decompose import decompose, edge_witnesses, interaction_graph, reach_set

ALPHABET = ("a", "b", "c", "d", "e")

SCHEDULE = st.lists(st.sampled_from(ALPHABET), min_size=0, max_size=8)
SCHEDULES = st.lists(SCHEDULE, min_size=1, max_size=6)


@given(SCHEDULE)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_reach_contains_trajectory_cells(schedule: list[str]) -> None:
    cells = reach_set(schedule)
    for t, v in enumerate(schedule):
        assert (v, t) in cells


@given(SCHEDULE)
@settings(max_examples=200)
def test_reach_fills_anchor_intervals(schedule: list[str]) -> None:
    first_occ: dict[str, int] = {}
    last_occ: dict[str, int] = {}
    for t, v in enumerate(schedule):
        first_occ.setdefault(v, t)
        last_occ[v] = t
    cells = reach_set(schedule)
    for v, a in first_occ.items():
        b = last_occ[v]
        for t in range(a, b + 1):
            assert (v, t) in cells, (schedule, v, t)


@given(SCHEDULES)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_every_edge_has_a_witness_cell(schedules: list[list[str]]) -> None:
    _n, edges = interaction_graph(schedules)
    witnesses = edge_witnesses(schedules)
    assert set(witnesses) == edges


@given(SCHEDULES)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_decompose_partitions_agents(schedules: list[list[str]]) -> None:
    result = decompose(schedules)
    n = len(schedules)
    covered: set[int] = set(result.singletons)
    for comp in result.non_trivial_components:
        assert covered.isdisjoint(comp), (result, comp, covered)
        covered |= set(comp)
    assert covered == set(range(n))
    assert (
        sum(1 for _ in result.singletons) + sum(len(c) for c in result.non_trivial_components) == n
    )


@given(SCHEDULES)
@settings(max_examples=200)
def test_component_of_consistent_with_singletons_and_components(
    schedules: list[list[str]],
) -> None:
    result = decompose(schedules)
    n = len(schedules)
    sizes: dict[int, int] = {}
    for i in range(n):
        idx = result.component_of[i]
        sizes[idx] = sizes.get(idx, 0) + 1
    histogram = result.component_size_histogram()
    sizes_in_histogram: dict[int, int] = {}
    for size, count in histogram.items():
        sizes_in_histogram[size] = sizes_in_histogram.get(size, 0) + count
    distribution = {1: result.n_singletons}
    for comp in result.non_trivial_components:
        distribution[len(comp)] = distribution.get(len(comp), 0) + 1
    assert distribution == histogram


@given(SCHEDULES)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_cross_component_reach_sets_are_disjoint(schedules: list[list[str]]) -> None:
    result = decompose(schedules)
    reaches = [reach_set(s) for s in schedules]
    singletons = set(result.singletons)
    nt_members = [set(c) for c in result.non_trivial_components]
    for i in range(len(schedules)):
        for j in range(i + 1, len(schedules)):
            same_singleton = i == j
            same_nt = any(i in mem and j in mem for mem in nt_members)
            same_component = (
                same_singleton or same_nt or (i in singletons and j in singletons and i == j)
            )
            if same_component or result.component_of[i] == result.component_of[j]:
                continue
            assert reaches[i].isdisjoint(reaches[j]), (schedules, i, j)


@given(SCHEDULES)
@settings(max_examples=200)
def test_no_self_edges(schedules: list[list[str]]) -> None:
    _n, edges = interaction_graph(schedules)
    for edge in edges:
        assert len(edge) == 2, edge
