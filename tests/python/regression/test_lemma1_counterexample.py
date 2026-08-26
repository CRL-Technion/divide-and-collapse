"""Counterexample to Lem. 1 (lem:dag-equivalence) of the paper.

Schedule ``['a','c','a','b','a','c','b']`` admits a cost-1 trajectory via the
non-maximal sub-loop ``Collapse(0, 2)`` for ``a``, but the maximal-only DP
emits cost 2 because the maximal collapse ``Collapse(0, 4)`` for ``a``
overlaps the disjoint collapse ``Collapse(3, 6)`` for ``b``.

The optimal trajectory:

    Indices  0  1  2  3  4  5  6
    Input    a  c  a  b  a  c  b
    Pi*      a  a  a  b  b  b  b

Uses collapse (0, 2) (pins 0..2 to 'a'), then keep (2, 3) is the unique move
'a' -> 'b' of cost 1, then collapse (3, 6) (pins 3..6 to 'b'). Total: 1 move.

This test asserts the brute-force all-arcs cost is 1. If our DP is fixed to
agree, the second assertion should be flipped to ``dp_cost == 1``.

Surfaced during Phase 1 hypothesis testing on 2026-06-21.
"""

from __future__ import annotations

from mapfc.single_agent.collapse_dag import baseline_cost, solve_unconstrained

COUNTEREXAMPLE: list[str] = ["a", "c", "a", "b", "a", "c", "b"]
EXPECTED_OPTIMAL_COST = 1
EXPECTED_OPTIMAL_PLAN: list[str] = ["a", "a", "a", "b", "b", "b", "b"]


def all_arcs_dp(schedule: list[str]) -> int:
    """Reference: DP over the FULL Collapse DAG (every valid ``(a, t)`` arc)."""
    k = len(schedule) - 1
    if k <= 0:
        return 0
    INF = k + 1
    f = [INF] * (k + 1)
    f[0] = 0
    for t in range(1, k + 1):
        keep_cost = 0 if schedule[t - 1] == schedule[t] else 1
        best = f[t - 1] + keep_cost
        for a in range(t):
            if schedule[a] == schedule[t] and f[a] < best:
                best = f[a]
        f[t] = best
    return f[k]


def test_explicit_plan_achieves_cost_one() -> None:
    assert baseline_cost(EXPECTED_OPTIMAL_PLAN) == EXPECTED_OPTIMAL_COST


def test_all_arcs_dp_returns_one() -> None:
    assert all_arcs_dp(COUNTEREXAMPLE) == EXPECTED_OPTIMAL_COST


def test_smart_sweep_recovers_optimum() -> None:
    """After the algorithm switch (Phase 1, 2026-06-21), the smart sweep on the
    full DAG handles this counterexample correctly."""
    dp_cost, plan = solve_unconstrained(COUNTEREXAMPLE)
    assert dp_cost == EXPECTED_OPTIMAL_COST
    assert plan == EXPECTED_OPTIMAL_PLAN
