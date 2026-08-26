"""Unconstrained Collapse DAG + linear-time solver for the 1-MAPFC primitive.

Given a single-agent input schedule ``M_i = (v_0, v_1, ..., v_k)``, the full
Collapse DAG ``D_i = (N_i, A_i)`` has two arc kinds:

- *keep arcs* ``(t, t+1)``: cost ``1`` if ``v_t != v_{t+1}`` else ``0``;
- *collapse arcs* ``(a, b)`` for every pair with ``M_i[a] == M_i[b]``,
  ``a < b``: cost ``0``.

In the worst case the collapse arcs are ``O(k_i^2)``. :func:`solve_unconstrained`
nonetheless runs in ``O(k_i)`` via a **smart sweep**: at each step we relax
only the single best incoming collapse arc, kept fresh in a per-vertex hash
``best_by_vertex[v]`` that points to the prior index with minimum ``dp`` value
among all ``a`` with ``M_i[a] == v``. The smart sweep is equivalent to a full
topological-order relaxation of every collapse arc, but does ``O(1)`` work
per timestep.

History: an earlier version of this module reduced the DAG to the
"maximal-loop" arcs ``(first(v), last(v))`` only, following Alg. 1 of an
earlier draft of the paper. That reduction is **not** cost-preserving — a
non-maximal sub-loop is sometimes strictly better because the maximal loop
overlaps a disjoint collapse at a different anchor and forces it to be
dropped. See ``tests/python/regression/test_lemma1_counterexample.py`` for
the minimal counterexample ``M_i = (a, c, a, b, a, c, b)`` (cost ``1`` via
sub-loops, cost ``2`` via maximal-only). The smart sweep here works directly
on the full DAG and never materialises the wrong reduction.

Public API:

- :func:`solve_unconstrained` — return ``(cost, plan)`` for the optimum.
- :func:`baseline_cost` — cost of the input schedule, used as an upper bound.
- :func:`maximal_loops` — diagnostic helper; emits the maximal-loop arcs of
  the input schedule. Kept for reporting and figure-generation purposes;
  ``solve_unconstrained`` does not consume it.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

V = TypeVar("V", bound=Hashable)
"""Vertex type. For POGEMA grids: ``tuple[int, int]``. Algorithms only require
equality and hashing; the choice of representation is the caller's."""


def baseline_cost(schedule: Sequence[V]) -> int:
    """Cost of ``schedule`` as-is: number of move transitions.

    A schedule of length ``k+1`` has ``k`` transitions; each transition is a
    *move* iff ``v_t != v_{t+1}``.
    """
    return sum(1 for t in range(len(schedule) - 1) if schedule[t] != schedule[t + 1])


def maximal_loops(schedule: Sequence[V]) -> list[tuple[int, int]]:
    """Enumerate the collapse arcs of the unconstrained Collapse DAG.

    Returns one ``(first, last)`` pair per anchor vertex visited at least
    twice (Lem. 2). Runs in ``O(k_i)`` time via a single pass that records
    each vertex's first and last occurrence.

    Output order: by ``first`` ascending, then ``last`` ascending. The order
    has no semantic effect but the determinism is convenient for tests.
    """
    first_occ: dict[V, int] = {}
    last_occ: dict[V, int] = {}
    for t, v in enumerate(schedule):
        if v not in first_occ:
            first_occ[v] = t
        last_occ[v] = t
    loops = [(first_occ[v], last_occ[v]) for v in first_occ if last_occ[v] > first_occ[v]]
    loops.sort()
    return loops


def solve_unconstrained(schedule: Sequence[V]) -> tuple[int, list[V]]:
    """Return ``(optimal cost, reconstructed pi*)`` for the unconstrained 1-MAPFC.

    Implements the smart sweep over the full Collapse DAG. The DP recurrence is

    ``dp[0] = 0``,
    ``dp[t] = min( dp[t-1] + keep_cost(t-1, t),
                   min_{a < t, M_i[a] == M_i[t]} dp[a] )``

    where ``keep_cost(t-1, t) = 0`` if waiting, ``1`` if moving. The inner
    ``min`` over ``a`` is maintained incrementally in ``best_by_vertex[v]``:
    the index ``a <= t-1`` with ``M_i[a] == v`` that minimises ``dp[a]``.
    Each timestep does ``O(1)`` work, so the total is ``O(k_i)``.

    ``pi*`` is reconstructed by walking the parent pointers from ``t = k``
    back to ``t = 0``; on a collapse-arc step ``(a, t)`` the trajectory is
    pinned to the anchor ``M_i[a]`` over the whole interval ``[a, t]``.
    """
    k = len(schedule) - 1
    if k <= 0:
        # Empty schedule (k = -1) or one-vertex schedule (k = 0): nothing to optimise.
        return 0, list(schedule)

    INF = k + 1  # any valid cost is at most k, so INF as a sentinel
    dp: list[int] = [INF] * (k + 1)
    parent: list[tuple[int, str] | None] = [None] * (k + 1)
    dp[0] = 0
    best_by_vertex: dict[V, int] = {schedule[0]: 0}

    for t in range(1, k + 1):
        keep_cost = 0 if schedule[t - 1] == schedule[t] else 1
        best = dp[t - 1] + keep_cost
        best_parent: tuple[int, str] = (t - 1, "keep")
        a = best_by_vertex.get(schedule[t])
        if a is not None and dp[a] < best:
            best = dp[a]
            best_parent = (a, "collapse")
        dp[t] = best
        parent[t] = best_parent

        v = schedule[t]
        prev_best = best_by_vertex.get(v)
        if prev_best is None or dp[t] < dp[prev_best]:
            best_by_vertex[v] = t

    plan = list(schedule)
    t = k
    while t > 0:
        p = parent[t]
        assert p is not None
        prev, kind = p
        if kind == "collapse":
            anchor = schedule[prev]
            for s in range(prev, t + 1):
                plan[s] = anchor
        t = prev

    return dp[k], plan
