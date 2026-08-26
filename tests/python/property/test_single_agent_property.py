"""Cross-validate the unconstrained 1-MAPFC DP against two reference solvers.

The two reference solvers test orthogonal properties:

- :func:`brute_force_subsets` enumerates every disjoint-or-touching subset of
  valid collapse intervals on schedules of length at most 5 (so the worst case
  is roughly :math:`2^{10}` masks). Confirms that the DP finds the true
  minimum cost, not just a local optimum.

- :func:`all_arcs_dp` runs the same shortest-path DP but over the *full* arc
  set (every ``(a, t)`` with ``M[a] == M[t]``, not only the maximal
  ``(first(v), last(v))`` pairs). Equality with our DP empirically corroborates
  Lem. 2 of the paper (maximal-loop dominance): if a sub-loop ever strictly
  improved on its maximal parent, the all-arcs DP would beat ours.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mapfc.single_agent.collapse_dag import baseline_cost, solve_unconstrained

ALPHABET = ("a", "b", "c")

SHORT_SCHED = st.lists(st.sampled_from(ALPHABET), min_size=0, max_size=5)
MED_SCHED = st.lists(st.sampled_from(ALPHABET), min_size=0, max_size=12)


def brute_force_subsets(schedule: list[str]) -> int:
    """Min cost over disjoint-or-endpoint-touching subsets of valid intervals."""
    k = len(schedule) - 1
    if k <= 0:
        return 0

    intervals: list[tuple[int, int]] = [
        (a, b) for a in range(k) for b in range(a + 1, k + 1) if schedule[a] == schedule[b]
    ]
    n = len(intervals)
    best = baseline_cost(schedule)

    for mask in range(1 << n):
        chosen = sorted(intervals[i] for i in range(n) if mask & (1 << i))
        if any(chosen[i + 1][0] < chosen[i][1] for i in range(len(chosen) - 1)):
            continue
        trajectory = list(schedule)
        for a, b in chosen:
            anchor = schedule[a]
            for t in range(a, b + 1):
                trajectory[t] = anchor
        c = baseline_cost(trajectory)
        if c < best:
            best = c

    return best


def all_arcs_dp(schedule: list[str]) -> int:
    """Same DP as solve_unconstrained but considers EVERY (a, t) collapse arc."""
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


@given(SHORT_SCHED)
@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
def test_dp_matches_subset_enumeration(schedule: list[str]) -> None:
    dp_cost, _ = solve_unconstrained(schedule)
    assert dp_cost == brute_force_subsets(schedule), schedule


@given(MED_SCHED)
@settings(max_examples=500)
def test_maximal_only_matches_all_arcs(schedule: list[str]) -> None:
    """Lem. 2: optimal cost using only maximal arcs equals cost using all arcs."""
    dp_cost, _ = solve_unconstrained(schedule)
    assert dp_cost == all_arcs_dp(schedule), schedule


@given(MED_SCHED)
@settings(max_examples=300)
def test_cost_at_most_baseline(schedule: list[str]) -> None:
    dp_cost, _ = solve_unconstrained(schedule)
    assert dp_cost <= baseline_cost(schedule), schedule


@given(MED_SCHED)
@settings(max_examples=300)
def test_plan_endpoints_match_input(schedule: list[str]) -> None:
    if not schedule:
        return
    _, plan = solve_unconstrained(schedule)
    assert plan[0] == schedule[0]
    assert plan[-1] == schedule[-1]


@given(MED_SCHED)
@settings(max_examples=300)
def test_plan_cost_consistent_with_reconstruction(schedule: list[str]) -> None:
    dp_cost, plan = solve_unconstrained(schedule)
    assert baseline_cost(plan) == dp_cost, (schedule, plan)
