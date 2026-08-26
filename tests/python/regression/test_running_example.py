"""Regression test on the running example of the paper.

The setup (Ex. ex:setup of single_agent.tex / formulation.tex):
- Star graph centred at ``b`` with arms ``a``, ``c``, ``d``, ``e``.
- Three agents with the following input schedules and expected per-agent
  optima (from Sec. 3 of the paper, Ex. ex:single-agent):

  +-------+---------------------+-------------------------+----------------------+
  | Agent | M_i                 | pi*_i                   | OPT(\\{i\\})        |
  +=======+=====================+=========================+======================+
  | 1     | <a, b, c, b, e>     | <a, b, b, b, e>         | 2                    |
  +-------+---------------------+-------------------------+----------------------+
  | 2     | <b, a, b, a, b>     | <b, b, b, b, b>         | 0                    |
  +-------+---------------------+-------------------------+----------------------+
  | 3     | <d, d, d, d, d>     | <d, d, d, d, d>         | 0                    |
  +-------+---------------------+-------------------------+----------------------+

The sum is the ``IndepLB`` of Sec. 6 (Experimental Evaluation): a lower bound
on the global MAPF-Collapse optimum obtained by solving each agent's
unconstrained sub-problem in isolation. Here, IndepLB = 2.

The example also doubles as the empirical sanity check for the smart-sweep
implementation: none of the three schedules exercises the Lem. 1
counterexample pattern (which requires three distinct vertices with
overlapping maximal loops at different anchors), so the maximal-only and
smart-sweep algorithms agree on each. The test still catches regressions
on the post-revision implementation.
"""

from __future__ import annotations

from mapfc.single_agent.collapse_dag import baseline_cost, solve_unconstrained
from mapfc.single_agent.constrained_dag import solve_constrained

AGENT_1_SCHEDULE = ["a", "b", "c", "b", "e"]
AGENT_2_SCHEDULE = ["b", "a", "b", "a", "b"]
AGENT_3_SCHEDULE = ["d", "d", "d", "d", "d"]

EXPECTED_PLAN_1 = ["a", "b", "b", "b", "e"]
EXPECTED_PLAN_2 = ["b", "b", "b", "b", "b"]
EXPECTED_PLAN_3 = ["d", "d", "d", "d", "d"]

EXPECTED_OPT_1 = 2
EXPECTED_OPT_2 = 0
EXPECTED_OPT_3 = 0

INDEP_LB = EXPECTED_OPT_1 + EXPECTED_OPT_2 + EXPECTED_OPT_3
ASSERTED_INDEP_LB = 2


def test_baselines_match_paper_text() -> None:
    """cost(M^1) = 4 (four moves a->b->c->b->e); M^2 = 4; M^3 = 0."""
    assert baseline_cost(AGENT_1_SCHEDULE) == 4
    assert baseline_cost(AGENT_2_SCHEDULE) == 4
    assert baseline_cost(AGENT_3_SCHEDULE) == 0


def test_agent_1_unconstrained() -> None:
    cost, plan = solve_unconstrained(AGENT_1_SCHEDULE)
    assert cost == EXPECTED_OPT_1
    assert plan == EXPECTED_PLAN_1


def test_agent_2_unconstrained() -> None:
    cost, plan = solve_unconstrained(AGENT_2_SCHEDULE)
    assert cost == EXPECTED_OPT_2
    assert plan == EXPECTED_PLAN_2


def test_agent_3_unconstrained() -> None:
    cost, plan = solve_unconstrained(AGENT_3_SCHEDULE)
    assert cost == EXPECTED_OPT_3
    assert plan == EXPECTED_PLAN_3


def test_indep_lb_matches_paper() -> None:
    assert INDEP_LB == ASSERTED_INDEP_LB


def test_constrained_matches_unconstrained_on_running_example() -> None:
    """Empty C_i: the constrained sweep must return the same answer."""
    for schedule, expected_opt, expected_plan in [
        (AGENT_1_SCHEDULE, EXPECTED_OPT_1, EXPECTED_PLAN_1),
        (AGENT_2_SCHEDULE, EXPECTED_OPT_2, EXPECTED_PLAN_2),
        (AGENT_3_SCHEDULE, EXPECTED_OPT_3, EXPECTED_PLAN_3),
    ]:
        result = solve_constrained(schedule, ())
        assert result is not None
        cost, plan = result
        assert cost == expected_opt
        assert plan == expected_plan
