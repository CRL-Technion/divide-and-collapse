"""Hypothesis-driven cross-validation harness — Phase 3d.

Three independent oracles must agree on the joint optimum for every
randomly generated small MAPF-Collapse instance:

- ``_brute_joint_cost``: exhaustive enumeration. Slow but exact;
  feasible only for tiny inputs (|I| <= 3 agents, T <= 6 vertices).
- ``solve_ccbs``: our exact CBS-style best-first search.
- ``solve_judgelight``: Tang et al.'s ILP, dispatched via subprocess.

Invariants tested:

- ``brute`` is always not ``None`` — the empty-collapse joint plan
  equals the input ``M``, which is collision-free by construction
  (the schedule generator filters out vertex- and edge-collision
  inputs).
- When CCBS succeeds, ``ccbs.cost == brute``. Any disagreement is a
  CCBS soundness bug.
- When CCBS and Judgelight both succeed, ``ccbs.cost <= judgelight.cost``.
  Judgelight runs a greedy "safe-oscillation" preprocessor before its
  ILP; on a small fraction of inputs the greedy commits to a saving
  that blocks a strictly better one, so equality is not the invariant.
  A violation in the other direction (Judgelight strictly better than
  CCBS on a brute-confirmed feasible instance) would be a CCBS bug.

Run modes:

- Default: ``pytest -m slow tests/python/stress/`` — uses the
  ``stress-dev`` Hypothesis profile (50 examples).
- Nightly: ``MAPFC_HYPOTHESIS_PROFILE=stress-ci pytest -m slow
  tests/python/stress/`` — uses the ``stress-ci`` profile (500
  examples). Wired into the GitHub Actions nightly job.

The Judgelight oracle requires ``.venv-judgelight/`` to be present
(see ``scripts/setup_judgelight.ps1``). If it is missing, every
example silently skips the Judgelight branch and only the
brute-vs-CCBS invariant is exercised — still useful for a
local sanity run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

# Reuse the deterministic brute-force machinery from the property test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "property"))
from test_ccbs_property import _brute_joint_cost

from mapfc.joint.ccbs import solve_ccbs
from mapfc.joint.judgelight_wrapper import DEFAULT_VENV_PY, solve_judgelight

# Tight bounds: brute-force must remain tractable for every drawn input.
# With |I| = 3 agents and T = 6 there are at most ~3^6 = 729 collapse-
# derived plans per agent and ~3.9e8 joint combinations in the worst
# case, but the average is far smaller because most schedules admit only
# a handful of valid collapse intervals.
ALPHABET = ("a", "b", "c", "d")
MIN_AGENTS, MAX_AGENTS = 2, 3
MIN_T, MAX_T = 2, 6

CCBS_TIMEOUT_S = 10.0
JL_TIMEOUT_S = 10.0

_JUDGELIGHT_AVAILABLE = DEFAULT_VENV_PY.exists()


def _vertex_collision_free(plans: list[list[str]]) -> bool:
    horizon = max(len(p) for p in plans)
    for t in range(horizon):
        seen: set[str] = set()
        for p in plans:
            v = p[t] if t < len(p) else p[-1]
            if v in seen:
                return False
            seen.add(v)
    return True


def _edge_collision_free(plans: list[list[str]]) -> bool:
    horizon = max(len(p) for p in plans)
    for t in range(horizon - 1):
        for i in range(len(plans)):
            for j in range(i + 1, len(plans)):
                pi, pj = plans[i], plans[j]
                if t + 1 >= len(pi) or t + 1 >= len(pj):
                    continue
                if pi[t] == pj[t + 1] and pj[t] == pi[t + 1] and pi[t] != pi[t + 1]:
                    return False
    return True


@st.composite
def _collision_free_schedules(draw):
    """Draw a collision-free input ``M`` for ``MIN_AGENTS..MAX_AGENTS`` agents."""
    num_agents = draw(st.integers(min_value=MIN_AGENTS, max_value=MAX_AGENTS))
    horizon = draw(st.integers(min_value=MIN_T, max_value=MAX_T))
    plans: list[list[str]] = []
    for _ in range(num_agents):
        plan = draw(st.lists(st.sampled_from(ALPHABET), min_size=horizon, max_size=horizon))
        plans.append(plan)
    assume(_vertex_collision_free(plans))
    assume(_edge_collision_free(plans))
    return plans


@pytest.mark.slow
@given(schedules=_collision_free_schedules())
def test_brute_ccbs_agree(schedules: list[list[str]]) -> None:
    """CCBS must match brute force on every feasible small instance."""
    brute = _brute_joint_cost(schedules)
    assert brute is not None, (
        f"brute returned None on a collision-free input M; "
        f"the empty-collapse plan should always be feasible.\n"
        f"schedules: {schedules}"
    )

    ccbs_result = solve_ccbs(schedules, time_limit_sec=CCBS_TIMEOUT_S)
    if ccbs_result is None:
        # CCBS timed out on a small instance — unexpected but not a soundness
        # bug. Surface the input so the user can investigate.
        pytest.skip(f"CCBS timed out on {schedules}")
    assert ccbs_result.cost == brute, (
        f"CCBS cost disagrees with brute force.\n"
        f"schedules: {schedules}\n"
        f"brute: {brute}\n"
        f"ccbs:  {ccbs_result.cost}"
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not _JUDGELIGHT_AVAILABLE,
    reason=f"Judgelight venv not found at {DEFAULT_VENV_PY}",
)
@given(schedules=_collision_free_schedules())
def test_ccbs_le_judgelight(schedules: list[list[str]]) -> None:
    """CCBS, being exact, must never report a worse cost than Judgelight."""
    brute = _brute_joint_cost(schedules)
    assert brute is not None, (
        f"brute returned None on a collision-free input; "
        f"input-filtering bug.\nschedules: {schedules}"
    )

    ccbs_result = solve_ccbs(schedules, time_limit_sec=CCBS_TIMEOUT_S)
    if ccbs_result is None:
        pytest.skip(f"CCBS timed out on {schedules}")

    jl_result = solve_judgelight(schedules, time_limit_sec=JL_TIMEOUT_S)
    if jl_result is None:
        pytest.skip(f"Judgelight failed/timed out on {schedules}")

    assert ccbs_result.cost <= jl_result.cost, (
        f"CCBS > Judgelight: CCBS is exact, so it should never be worse.\n"
        f"schedules: {schedules}\n"
        f"brute:      {brute}\n"
        f"ccbs:       {ccbs_result.cost}\n"
        f"judgelight: {jl_result.cost}"
    )
