"""Per-agent reach set Reach^i of the decomposition pipeline (Sec. 4 of the paper).

The reach set is the union of the *trajectory cells* the agent occupies on its
input schedule and the *anchor cells* the agent could occupy under any
single-agent plan derived from the schedule by collapse operations:

    Reach^i = Reach^i_orig union Reach^i_anchor

with

    Reach^i_orig    = { (M^i(t), t) : 0 <= t <= k_i }
    Reach^i_anchor  = { (v, t) : v visited at least twice by M^i,
                                 first_i(v) <= t <= last_i(v) }

The original cells are a subset of the anchor cells only at indices where the
agent's trajectory vertex is the anchor of some maximal loop; in general the
union of the two adds the "covered times" of every multi-visit vertex.

Used by :mod:`mapfc.decompose.interaction` to build the interaction graph H.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

V = TypeVar("V", bound=Hashable)


def reach_set(schedule: Sequence[V]) -> set[tuple[V, int]]:
    """Return Reach^i for a single agent's schedule M^i.

    The trajectory cells ``(M^i(t), t)`` are emitted first; then, for every
    vertex visited at least twice, the entire ``[first, last]`` interval is
    filled with anchor cells ``(v, t)``. Set semantics absorbs duplicates so
    multi-anchor cells (e.g., a vertex visited every step) are recorded once.

    Time: ``O(k_i + sum over multi-visit anchors of interval length)``.
    Space: same.
    """
    cells: set[tuple[V, int]] = set()
    first_occ: dict[V, int] = {}
    last_occ: dict[V, int] = {}
    for t, v in enumerate(schedule):
        cells.add((v, t))
        if v not in first_occ:
            first_occ[v] = t
        last_occ[v] = t
    for v, a in first_occ.items():
        b = last_occ[v]
        if b > a:
            for t in range(a, b + 1):
                cells.add((v, t))
    return cells
