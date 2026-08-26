"""Decomposition pipeline (Alg. 2 of method.tex): schedules -> components.

Given a MAPF-Collapse instance ``(G, M)`` with ``M = (M^1, ..., M^N)``,
:func:`decompose` builds the per-agent reach sets, the interaction graph H,
and its connected components, partitioning the result into:

- ``singletons``    : agent indices whose component has size 1.
- ``non_trivial_components`` : frozensets of agent indices for components of
  size 2 or more.

Plus measurement fields ``edges_count`` and ``h_construction_time_s`` that
feed into Q2 of the experimental evaluation.
"""

from __future__ import annotations

import time
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from mapfc.decompose.components import connected_components
from mapfc.decompose.interaction import interaction_graph
from mapfc.decompose.reach import reach_set

V = TypeVar("V", bound=Hashable)


@dataclass(frozen=True)
class DecompositionResult:
    """Output of :func:`decompose`."""

    n_agents: int
    singletons: tuple[int, ...]
    non_trivial_components: tuple[frozenset[int], ...]
    component_of: tuple[int, ...]
    edges_count: int
    reach_total_cells: int
    h_construction_time_s: float

    @property
    def n_components(self) -> int:
        return len(self.singletons) + len(self.non_trivial_components)

    @property
    def n_singletons(self) -> int:
        return len(self.singletons)

    @property
    def singleton_fraction(self) -> float:
        return self.n_singletons / self.n_agents if self.n_agents else 0.0

    @property
    def max_non_trivial_size(self) -> int:
        if not self.non_trivial_components:
            return 0
        return max(len(c) for c in self.non_trivial_components)

    def component_size_histogram(self) -> dict[int, int]:
        hist: dict[int, int] = {1: self.n_singletons}
        for comp in self.non_trivial_components:
            hist[len(comp)] = hist.get(len(comp), 0) + 1
        return {k: hist[k] for k in sorted(hist)}


def decompose(schedules: Sequence[Sequence[V]]) -> DecompositionResult:
    """Apply Alg. 2 of the paper to ``schedules`` and return the decomposition.

    Wall-clock measurement covers reach-set construction, interaction graph
    construction, and connected-component computation. Plan reconstruction
    (per-agent solves) is downstream and out of scope here.
    """
    t0 = time.perf_counter()

    reach_sets: list[set[tuple[V, int]]] = [reach_set(s) for s in schedules]
    reach_total = sum(len(r) for r in reach_sets)

    n = len(schedules)
    cell_to_agents: dict[tuple[V, int], list[int]] = {}
    for i, reach in enumerate(reach_sets):
        for cell in reach:
            cell_to_agents.setdefault(cell, []).append(i)
    edges: set[frozenset[int]] = set()
    for agents in cell_to_agents.values():
        if len(agents) < 2:
            continue
        for x in range(len(agents)):
            for y in range(x + 1, len(agents)):
                edges.add(frozenset({agents[x], agents[y]}))

    components, component_of_arr = connected_components(n, edges)
    elapsed = time.perf_counter() - t0

    singletons: list[int] = []
    non_trivial: list[frozenset[int]] = []
    for comp in components:
        if len(comp) == 1:
            singletons.append(next(iter(comp)))
        else:
            non_trivial.append(comp)

    return DecompositionResult(
        n_agents=n,
        singletons=tuple(sorted(singletons)),
        non_trivial_components=tuple(non_trivial),
        component_of=component_of_arr,
        edges_count=len(edges),
        reach_total_cells=reach_total,
        h_construction_time_s=elapsed,
    )


__all__ = [
    "DecompositionResult",
    "connected_components",
    "decompose",
    "interaction_graph",
    "reach_set",
]
