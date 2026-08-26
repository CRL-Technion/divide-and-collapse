"""Hand-crafted edge cases for mapfc.joint.ccbs.solve_ccbs."""

from __future__ import annotations

from mapfc.joint import baseline_cost
from mapfc.joint.ccbs import Conflict, find_vertex_conflict, solve_ccbs


def test_empty_schedules() -> None:
    result = solve_ccbs([])
    assert result is not None
    assert result.cost == 0
    assert result.plans == ()
    assert result.solver == "ccbs"


def test_single_agent_matches_smart_sweep() -> None:
    """Component of size 1 is conflict-free at the root."""
    schedules = [("a", "b", "c", "b", "d")]
    result = solve_ccbs(schedules)
    assert result is not None
    assert result.cost == 2
    assert result.plans == (("a", "b", "b", "b", "d"),)


def test_two_disjoint_agents() -> None:
    """No shared vertex => no conflict => root returned with summed cost."""
    schedules = [("a", "b", "a"), ("c", "d", "c")]
    result = solve_ccbs(schedules)
    assert result is not None
    assert result.cost == 0
    assert result.plans[0] == ("a", "a", "a")
    assert result.plans[1] == ("c", "c", "c")


def test_running_example_two_agents() -> None:
    """Paper's running example, agents 1 and 2 in the non-trivial component.

    The unconstrained per-agent optima are:
      agent 0: cost 2, plan ("a","b","b","b","e")
      agent 1: cost 0, plan ("b","b","b","b","b")
    These collide at b at times 1..3. CCBS resolves by constraining one
    agent away from (b, 1) (the earliest conflict). Constraining agent 0
    makes it infeasible (the trajectory itself sits at b at t=1), so the
    surviving branch constrains agent 1, whose constrained optimum is
    cost 2 (plan ("b","a","a","a","b")). Joint cost: 2 + 2 = 4.
    """
    schedules = [("a", "b", "c", "b", "e"), ("b", "a", "b", "a", "b")]
    result = solve_ccbs(schedules)
    assert result is not None
    assert result.cost == 4
    assert result.plans[0] == ("a", "b", "b", "b", "e")
    assert result.plans[1] == ("b", "a", "a", "a", "b")
    assert find_vertex_conflict(result.plans) is None


def test_returned_plan_has_no_conflict() -> None:
    """Agents that share no vertex; root is conflict-free."""
    schedules = [("a", "b", "a", "b", "a"), ("c", "d", "c", "d", "c")]
    result = solve_ccbs(schedules)
    assert result is not None
    assert find_vertex_conflict(result.plans) is None


def test_find_vertex_conflict_no_conflict() -> None:
    plans = [("a", "b"), ("c", "d")]
    assert find_vertex_conflict(plans) is None


def test_find_vertex_conflict_simple() -> None:
    plans = [("a", "b"), ("a", "c")]
    conflict = find_vertex_conflict(plans)
    assert conflict == Conflict(i=0, j=1, v="a", t=0)


def test_find_vertex_conflict_earliest_time_wins() -> None:
    plans = [("a", "b", "c"), ("a", "d", "c")]
    conflict = find_vertex_conflict(plans)
    assert conflict is not None
    assert conflict.t == 0
    assert conflict.v == "a"


def test_find_vertex_conflict_padding() -> None:
    """Shorter trajectories pad with last vertex (wait)."""
    plans = [("a", "b"), ("c", "d", "b")]
    conflict = find_vertex_conflict(plans)
    assert conflict == Conflict(i=0, j=1, v="b", t=2)


def test_find_vertex_conflict_empty() -> None:
    assert find_vertex_conflict([]) is None


def test_time_limit_returns_none() -> None:
    """Negative budget guarantees the time check fires before the first pop."""
    schedules = [("a", "b", "c", "b", "e"), ("b", "a", "b", "a", "b")]
    result = solve_ccbs(schedules, time_limit_sec=-1.0)
    assert result is None


def test_plan_length_preserved() -> None:
    schedules = [("a", "b", "c", "b", "e"), ("b", "a", "b", "a", "b")]
    result = solve_ccbs(schedules)
    assert result is not None
    for plan, sched in zip(result.plans, schedules, strict=True):
        assert len(plan) == len(sched)


def test_cost_equals_baseline_cost_of_returned_plan() -> None:
    schedules = [("a", "b", "c", "b", "e"), ("b", "a", "b", "a", "b")]
    result = solve_ccbs(schedules)
    assert result is not None
    assert baseline_cost(result.plans) == result.cost


def test_grid_cell_vertex_type() -> None:
    """CCBS must work on tuple vertices (POGEMA grid cells); pick
    spatially-disjoint trajectories so M is collision-free."""
    schedules = [
        ((0, 0), (0, 1), (0, 0)),  # agent 0 oscillates near (0, 0)
        ((5, 5), (5, 6), (5, 5)),  # agent 1 oscillates far away
    ]
    result = solve_ccbs(schedules)
    assert result is not None
    assert result.cost == 0
    for v in result.plans[0]:
        assert isinstance(v, tuple)


def test_three_agent_chain() -> None:
    """Three agents on disjoint vertex sets => no conflict."""
    schedules = [("a", "b", "a"), ("c", "d", "c"), ("e", "f", "e")]
    result = solve_ccbs(schedules)
    assert result is not None
    assert result.cost == 0


def test_default_is_cardinal_first_from_root() -> None:
    """The default configuration classifies conflicts from the root
    (``cardinal_trigger_pops=0``); the deferred selector is opt-in."""
    schedules = [("a", "b", "c", "b", "e"), ("b", "a", "b", "a", "b")]
    diagnostics: dict[str, object] = {}
    result = solve_ccbs(schedules, diagnostics=diagnostics)
    assert result is not None
    assert result.cost == 4
    assert diagnostics["cardinal_activated"] is True


def test_deferred_selector_skips_cardinal_on_small_component() -> None:
    """With an explicit deferral threshold, a component that closes within
    the budget never activates the cardinal classifier."""
    schedules = [("a", "b", "c", "b", "e"), ("b", "a", "b", "a", "b")]
    diagnostics: dict[str, object] = {}
    result = solve_ccbs(schedules, cardinal_trigger_pops=16, diagnostics=diagnostics)
    assert result is not None
    assert result.cost == 4
    assert diagnostics["cardinal_activated"] is False
    assert int(diagnostics["pop_count"]) <= 16


def test_adaptive_threshold_zero_forces_cardinal() -> None:
    """``cardinal_trigger_pops=0`` flips cardinal-first on at the root."""
    schedules = [("a", "b", "c", "b", "e"), ("b", "a", "b", "a", "b")]
    diagnostics: dict[str, object] = {}
    result = solve_ccbs(schedules, cardinal_trigger_pops=0, diagnostics=diagnostics)
    assert result is not None
    assert result.cost == 4
    assert diagnostics["cardinal_activated"] is True


def test_disjoint_split_running_example() -> None:
    """Disjoint split (default) and vanilla branching agree on the running example."""
    schedules = [("a", "b", "c", "b", "e"), ("b", "a", "b", "a", "b")]
    djs = solve_ccbs(schedules, use_disjoint_split=True)
    vanilla = solve_ccbs(schedules, use_disjoint_split=False)
    assert djs is not None and vanilla is not None
    assert djs.cost == vanilla.cost == 4
    assert find_vertex_conflict(djs.plans) is None
    assert find_vertex_conflict(vanilla.plans) is None


def test_disjoint_split_three_way_conflict() -> None:
    """Three agents all colliding at the same cell: the positive constraint
    on agent i must propagate to BOTH j and k."""
    schedules = [
        ("a", "b", "a"),
        ("c", "b", "c"),
        ("d", "b", "d"),
    ]
    djs = solve_ccbs(schedules, use_disjoint_split=True)
    vanilla = solve_ccbs(schedules, use_disjoint_split=False)
    assert djs is not None and vanilla is not None
    assert djs.cost == vanilla.cost
    assert find_vertex_conflict(djs.plans) is None


def test_bypass_matches_no_bypass_on_running_example() -> None:
    """Bypass must not change the joint optimum on the paper's running example."""
    schedules = [("a", "b", "c", "b", "e"), ("b", "a", "b", "a", "b")]
    with_bypass = solve_ccbs(schedules, use_bypass=True)
    without_bypass = solve_ccbs(schedules, use_bypass=False)
    assert with_bypass is not None and without_bypass is not None
    assert with_bypass.cost == without_bypass.cost == 4
    assert find_vertex_conflict(with_bypass.plans) is None


def test_bypass_diagnostics_count_exposed() -> None:
    """``diagnostics['bypass_count']`` is populated on every return path."""
    schedules = [("a", "b", "c", "b", "e"), ("b", "a", "b", "a", "b")]
    diag: dict[str, object] = {}
    result = solve_ccbs(schedules, diagnostics=diag)
    assert result is not None
    assert "bypass_count" in diag
    assert int(diag["bypass_count"]) >= 0
