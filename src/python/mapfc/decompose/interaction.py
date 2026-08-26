"""Interaction graph H = (I, E) on the agents (Sec. 4 of the paper).

For a MAPF-Collapse instance ``(G, M)`` with agents ``I = {1, ..., N}``, two
agents are H-adjacent iff their reach sets share at least one cell:

    {i, j} in E  iff  Reach^i intersect Reach^j is non-empty.

The construction here is the natural single pass: build a cell -> agent-list
hash, then emit one pairwise edge per pair of agents sharing any cell. Edges
are returned as frozenset({i, j}) tuples to be set-comparable across runs.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from typing import TypeVar

from mapfc.decompose.reach import reach_set

V = TypeVar("V", bound=Hashable)


def interaction_graph(
    schedules: Sequence[Sequence[V]],
) -> tuple[int, set[frozenset[int]]]:
    """Return ``(n, edges)`` of the interaction graph H on ``n = len(schedules)``.

    Each edge is a 2-element ``frozenset({i, j})`` of distinct agent indices.
    The construction iterates each agent's reach set once and groups cells by
    ``(vertex, time)``; cells touched by at least two agents contribute the
    full pairwise edge set among those agents.

    Time: ``O(sum_i |Reach^i| + sum_c (|agents at c| choose 2))`` in the worst
    case. On POGEMA-like sparsity, the per-cell agent set is small and the
    cubic term collapses.
    """
    cell_to_agents: dict[tuple[V, int], list[int]] = {}
    for i, schedule in enumerate(schedules):
        for cell in reach_set(schedule):
            cell_to_agents.setdefault(cell, []).append(i)

    edges: set[frozenset[int]] = set()
    for agents in cell_to_agents.values():
        if len(agents) < 2:
            continue
        for x in range(len(agents)):
            ax = agents[x]
            for y in range(x + 1, len(agents)):
                ay = agents[y]
                if ax != ay:
                    edges.add(frozenset({ax, ay}))
    return len(schedules), edges


def edge_witnesses(
    schedules: Sequence[Sequence[V]],
) -> dict[frozenset[int], tuple[V, int]]:
    """Return one witness cell per edge of H.

    For each ``(i, j)`` edge, return a single ``(v, t)`` cell in
    ``Reach^i intersect Reach^j``. Useful for tracing why two agents are
    deemed interacting.
    """
    reaches: list[set[tuple[V, int]]] = [reach_set(s) for s in schedules]
    witnesses: dict[frozenset[int], tuple[V, int]] = {}
    for i in range(len(reaches)):
        for j in range(i + 1, len(reaches)):
            common = reaches[i] & reaches[j]
            if common:
                witnesses[frozenset({i, j})] = min(common, key=lambda c: (c[1], _repr_key(c[0])))
    return witnesses


def _repr_key(v: object) -> str:
    return repr(v)


def edges_from_reach_sets(
    reach_sets_by_agent: Iterable[set[tuple[V, int]]],
) -> tuple[int, set[frozenset[int]]]:
    """Same as :func:`interaction_graph` but consumes precomputed reach sets.

    Used by callers that already materialise ``Reach^i`` for other purposes
    (e.g., the Q2 reporter that wants both H and per-agent reach statistics).
    """
    cell_to_agents: dict[tuple[V, int], list[int]] = {}
    n = 0
    for i, reach in enumerate(reach_sets_by_agent):
        n = i + 1
        for cell in reach:
            cell_to_agents.setdefault(cell, []).append(i)

    edges: set[frozenset[int]] = set()
    for agents in cell_to_agents.values():
        if len(agents) < 2:
            continue
        for x in range(len(agents)):
            ax = agents[x]
            for y in range(x + 1, len(agents)):
                ay = agents[y]
                if ax != ay:
                    edges.add(frozenset({ax, ay}))
    return n, edges
