"""Constrained Collapse DAG + per-window smart sweep (Prop. 4.1 of the paper).

A joint solver issues *negative* reach constraints ``i in/(v, t)`` (the agent
must NOT occupy ``(v, t)``) and, under disjoint-splitting CBS (App. A.4),
also *positive* reach constraints ``i = (v, t)`` (the agent MUST occupy
``(v, t)``). Given a finite set of forbidden cells ``forbidden_cells`` and
an optional set of required cells ``required_cells`` for one agent, this
module solves the constrained single-agent 1-MAPFC:

    Find a min-cost trajectory derived from M_i by collapse operations such
    that for every time t, the agent's position is not a forbidden cell
    AND for every ``(v, t) in required_cells`` the agent's position at t is v.

The required-cell case is reduced to the forbidden-cell case: requiring v at
t is equivalent to forbidding every other vertex u != v at t, restricted to
vertices the agent could possibly be at (the distinct values of M_i).

Vocabulary (matching the paper):

- A node ``t`` is **blocked** if ``(M_i(t), t)`` is forbidden: the agent's own
  trajectory cell at time ``t`` is forbidden, so a plan that keeps the agent
  on its trajectory at time ``t`` violates the constraint.
- For each anchor vertex ``v`` visited at least twice by ``M_i``, a
  **forbidden time** of ``v`` is any ``t in [first_i(v), last_i(v)]`` with
  ``(v, t)`` forbidden.
- A **forbidden-free window** of ``v`` is a maximal index interval
  ``W subset [first_i(v), last_i(v)]`` containing no forbidden time of ``v``.

A collapse arc ``(a, b)`` at anchor ``v`` is admissible iff
``[a, b]`` lies inside some forbidden-free window of ``v``. Keep arcs out of
or into blocked nodes are inadmissible.

Algorithm: the constrained smart sweep of Prop. 4.1. At each step we maintain
an argmin hash ``B: V -> index`` such that ``B[v]`` is the index ``a <= t``
with ``M_i(a) = v`` minimising ``dp[a]`` *within the current window of v*. An
inverted index ``R[t] = {v : (v, t) in forbidden_cells}`` lets us locate
window boundaries: when the sweep crosses a forbidden time of ``v``, we erase
``B[v]`` before the relaxation so subsequent collapse-arc lookups for ``v``
reference an index strictly greater than ``t``. Erase-then-relax is the
critical ordering — the unconstrained smart sweep of
:func:`mapfc.single_agent.collapse_dag.solve_unconstrained` is exactly the
special case ``forbidden_cells = ()``.

Public API: :func:`solve_constrained` returns ``(cost, plan)`` or ``None`` if
no plan derived from ``M_i`` by collapse operations satisfies the
constraints.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from typing import TypeVar

V = TypeVar("V", bound=Hashable)


def solve_constrained(
    schedule: Sequence[V],
    forbidden_cells: Iterable[tuple[V, int]],
    required_cells: Iterable[tuple[V, int]] = (),
) -> tuple[int, list[V]] | None:
    """Return ``(cost, pi*)`` of the constrained 1-MAPFC, or ``None`` if infeasible.

    Parameters
    ----------
    schedule
        The input single-agent plan ``M_i = (v_0, ..., v_k)`` as a sequence
        of vertices.
    forbidden_cells
        Any iterable of ``(vertex, time)`` pairs the agent must avoid.
        Times outside ``[0, k]`` are silently ignored; duplicate pairs are
        coalesced.
    required_cells
        Any iterable of ``(vertex, time)`` pairs the agent must occupy
        (positive constraints, App. A.4). Requiring v at t is reduced to
        forbidding every other distinct value u of M_i at t — the agent's
        position at t can only be a vertex appearing in M_i, so the
        reduction is exact. If two required cells share a time t with
        distinct vertices, the instance is infeasible by definition and
        ``None`` is returned.

    Returns
    -------
    ``None``
        If no plan derived from ``schedule`` by a sequence of collapses
        satisfies every constraint.
    ``(cost, plan)``
        Otherwise, the minimum move count and the corresponding trajectory.
    """
    k = len(schedule) - 1
    if k < 0:
        return 0, []

    forbid_at: dict[int, set[V]] = {}
    for v, t in forbidden_cells:
        if 0 <= t <= k:
            forbid_at.setdefault(t, set()).add(v)

    required_at: dict[int, V] = {}
    for v, t in required_cells:
        if not (0 <= t <= k):
            continue
        if t in required_at and required_at[t] != v:
            return None  # contradictory positive constraints
        required_at[t] = v
    if required_at:
        distinct_vertices = set(schedule)
        for t, v_pos in required_at.items():
            if v_pos not in distinct_vertices:
                return None  # required vertex never appears in M_i at any time
            bucket = forbid_at.setdefault(t, set())
            for u in distinct_vertices:
                if u != v_pos:
                    bucket.add(u)

    if schedule[0] in forbid_at.get(0, ()):
        return None

    INF = k + 2
    dp: list[int] = [INF] * (k + 1)
    parent: list[tuple[int, str] | None] = [None] * (k + 1)
    B: dict[V, int] = {}

    dp[0] = 0
    B[schedule[0]] = 0

    for t in range(1, k + 1):
        forbid_t = forbid_at.get(t, ())
        for v in forbid_t:
            B.pop(v, None)

        if schedule[t] in forbid_t:
            dp[t] = INF
            continue

        if dp[t - 1] < INF:
            keep_cost = 0 if schedule[t - 1] == schedule[t] else 1
            dp[t] = dp[t - 1] + keep_cost
            parent[t] = (t - 1, "keep")

        v = schedule[t]
        if v in B and dp[B[v]] < dp[t]:
            dp[t] = dp[B[v]]
            parent[t] = (B[v], "collapse")

        if dp[t] < INF:
            prev = B.get(v)
            if prev is None or dp[t] < dp[prev]:
                B[v] = t

    if dp[k] >= INF:
        return None

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
