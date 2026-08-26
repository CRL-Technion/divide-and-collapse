"""Property tests for the constrained 1-MAPFC smart sweep.

Three cross-checks:

- :func:`test_empty_constraints_matches_unconstrained` confirms that on
  ``forbidden = ()`` the constrained solver agrees with the unconstrained
  smart sweep of :func:`mapfc.single_agent.collapse_dag.solve_unconstrained`
  on every schedule. This is the load-bearing cross-check that would have
  caught the original Lem. 1 bug if it had been in place.

- :func:`test_constrained_matches_brute_force` runs an independent O(k^3)
  brute-force constrained DP that enumerates every valid ``(a, t)`` collapse
  arc and explicitly tests window membership, then compares it against the
  smart sweep on randomly generated schedule + forbidden-cell instances.

- :func:`test_adding_constraint_does_not_help` tests monotonicity: adding a
  forbidden cell to an instance never strictly decreases the optimum cost,
  and an infeasible instance stays infeasible under further constraints.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mapfc.single_agent.collapse_dag import baseline_cost, solve_unconstrained
from mapfc.single_agent.constrained_dag import solve_constrained

ALPHABET = ("a", "b", "c")
MAX_LEN = 10

SCHEDULE = st.lists(st.sampled_from(ALPHABET), min_size=0, max_size=MAX_LEN)


@st.composite
def schedule_with_forbidden(draw: st.DrawFn) -> tuple[list[str], frozenset[tuple[str, int]]]:
    schedule = draw(SCHEDULE)
    if not schedule:
        return schedule, frozenset()
    cells = draw(
        st.sets(
            st.tuples(
                st.sampled_from(ALPHABET),
                st.integers(min_value=0, max_value=len(schedule) - 1),
            ),
            max_size=4,
        )
    )
    return schedule, frozenset(cells)


def brute_force_constrained(
    schedule: list[str], forbidden: frozenset[tuple[str, int]]
) -> int | None:
    """All-arcs constrained DP that explicitly checks window membership."""
    k = len(schedule) - 1
    if k < 0:
        return 0
    if (schedule[0], 0) in forbidden:
        return None
    INF = k + 2
    dp = [INF] * (k + 1)
    dp[0] = 0
    for t in range(1, k + 1):
        if (schedule[t], t) in forbidden:
            dp[t] = INF
            continue
        if dp[t - 1] < INF:
            keep_cost = 0 if schedule[t - 1] == schedule[t] else 1
            dp[t] = dp[t - 1] + keep_cost
        v = schedule[t]
        for a in range(t):
            if schedule[a] != v:
                continue
            if any((v, s) in forbidden for s in range(a, t + 1)):
                continue
            if dp[a] < dp[t]:
                dp[t] = dp[a]
    return dp[k] if dp[k] < INF else None


@given(SCHEDULE)
@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
def test_empty_constraints_matches_unconstrained(schedule: list[str]) -> None:
    constrained = solve_constrained(schedule, ())
    unconstrained = solve_unconstrained(schedule)
    assert constrained is not None
    assert constrained[0] == unconstrained[0], schedule
    assert constrained[1] == unconstrained[1], schedule


@given(schedule_with_forbidden())
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
def test_constrained_matches_brute_force(
    instance: tuple[list[str], frozenset[tuple[str, int]]],
) -> None:
    schedule, forbidden = instance
    smart = solve_constrained(schedule, forbidden)
    brute = brute_force_constrained(schedule, forbidden)
    if brute is None:
        assert smart is None, (schedule, forbidden)
    else:
        assert smart is not None, (schedule, forbidden)
        assert smart[0] == brute, (schedule, forbidden, smart, brute)


@given(schedule_with_forbidden(), st.tuples(st.sampled_from(ALPHABET), st.integers(0, MAX_LEN)))
@settings(max_examples=200)
def test_adding_constraint_does_not_help(
    instance: tuple[list[str], frozenset[tuple[str, int]]],
    extra: tuple[str, int],
) -> None:
    schedule, forbidden = instance
    base = solve_constrained(schedule, forbidden)
    stricter = solve_constrained(schedule, forbidden | {extra})
    if base is None:
        assert stricter is None, (schedule, forbidden, extra)
        return
    if stricter is None:
        return
    assert stricter[0] >= base[0], (schedule, forbidden, extra, base, stricter)


@given(schedule_with_forbidden())
@settings(max_examples=200)
def test_cost_never_exceeds_baseline_when_feasible(
    instance: tuple[list[str], frozenset[tuple[str, int]]],
) -> None:
    schedule, forbidden = instance
    result = solve_constrained(schedule, forbidden)
    if result is None:
        return
    assert result[0] <= baseline_cost(schedule), (schedule, forbidden)


@given(schedule_with_forbidden())
@settings(max_examples=200)
def test_plan_satisfies_constraints_when_feasible(
    instance: tuple[list[str], frozenset[tuple[str, int]]],
) -> None:
    """The reconstructed plan must never place the agent on a forbidden cell."""
    schedule, forbidden = instance
    result = solve_constrained(schedule, forbidden)
    if result is None:
        return
    _, plan = result
    for t, v in enumerate(plan):
        assert (v, t) not in forbidden, (schedule, forbidden, plan, t, v)
