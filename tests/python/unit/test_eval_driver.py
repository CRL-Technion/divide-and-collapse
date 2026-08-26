"""Driver-contract unit tests for mapfc.eval.run_config.

Pins the contract that every Q4 row produced by the driver respects:
- ``nocollapse`` returns the input cost and zero saving.
- ``indep_lb`` matches the per-singleton ``solve_unconstrained`` sum.
- ``mapfc+ccbs`` returns the optimum on the running example.
- timeouts and ``None`` returns are recorded with ``isr=False`` and a
  non-empty ``failure_reason``.
- ``build_runtime_s == 0`` for every non-Judgelight config.
- The decomposition diagnostics match the underlying
  :class:`DecompositionResult`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from mapfc.decompose import decompose
from mapfc.eval import CONFIG_BY_NAME, Configuration, run_config
from mapfc.joint.base import JointPlan, baseline_cost
from mapfc.single_agent.collapse_dag import solve_unconstrained

# Running example (paper Sec. 2): 3 agents on a small vertex set.
RUNNING_SCHEDULES = (
    ("a", "b", "c", "b", "e"),
    ("b", "a", "b", "a", "b"),
    ("c", "d", "e", "d", "c"),
)


def _run(cname: str, schedules=RUNNING_SCHEDULES, budget_s: float = 5.0):
    cfg = CONFIG_BY_NAME[cname]
    return run_config(
        cfg,
        dataset="test",
        scenario="running",
        num_agents=len(schedules),
        schedules=schedules,
        budget_s=budget_s,
    )


def test_nocollapse_returns_input_cost() -> None:
    rec, jp = _run("nocollapse")
    assert rec.soc == rec.input_soc
    assert rec.saved_soc == 0
    assert rec.saving_ratio == 0.0
    assert rec.isr is True
    assert rec.decomp_runtime_s == 0.0
    assert rec.singleton_dispatch_s == 0.0
    assert rec.joint_solver_s == 0.0
    assert rec.failure_reason == ""
    assert jp is not None and jp.cost == rec.soc


def test_indep_lb_matches_per_singleton_sum() -> None:
    rec, _jp = _run("indep_lb")
    expected = sum(solve_unconstrained(s)[0] for s in RUNNING_SCHEDULES)
    assert rec.soc == expected
    assert rec.saved_soc == rec.input_soc - expected
    assert rec.isr is True
    # indep_lb does no decomposition, so the runtime_solver_only_s mirrors
    # singleton_dispatch_s.
    assert rec.runtime_solver_only_s == rec.singleton_dispatch_s
    assert rec.failure_reason == ""


def test_input_soc_matches_baseline_cost() -> None:
    rec, _ = _run("nocollapse")
    assert rec.input_soc == baseline_cost(RUNNING_SCHEDULES)


def test_mapfc_ccbs_running_example_cost_is_four() -> None:
    """The paper's running-example cost is 4 (Sec. 2)."""
    rec, jp = _run("mapfc+ccbs")
    assert rec.soc == 4
    assert rec.failure_reason == ""
    assert rec.isr is True
    assert jp is not None and jp.cost == 4


def test_decomp_diagnostics_match_decomposition_result() -> None:
    decomp = decompose(list(RUNNING_SCHEDULES))
    rec, _ = _run("mapfc+ccbs")
    assert rec.singleton_count == decomp.n_singletons
    assert rec.nontrivial_count == len(decomp.non_trivial_components)
    assert rec.max_nontrivial_size == decomp.max_non_trivial_size
    assert rec.singleton_fraction == decomp.singleton_fraction
    assert rec.edges_count == decomp.edges_count


def test_build_runtime_zero_for_ours() -> None:
    for cname in ("nocollapse", "indep_lb", "mapfc+ccbs"):
        rec, _ = _run(cname)
        assert rec.build_runtime_s == 0.0, cname


def test_mapfc_ccbs_recomposes_all_agents() -> None:
    _rec, jp = _run("mapfc+ccbs")
    assert jp is not None
    assert len(jp.plans) == len(RUNNING_SCHEDULES)
    for plan, sched in zip(jp.plans, RUNNING_SCHEDULES, strict=True):
        assert len(plan) == len(sched)


def test_disjoint_schedules_are_all_singletons() -> None:
    """When no two agents share a vertex, the decomposition is all singletons."""
    schedules = (
        ("a", "b", "a"),
        ("c", "d", "c"),
        ("e", "f", "e"),
    )
    rec, jp = _run("mapfc+ccbs", schedules=schedules)
    assert rec.singleton_count == 3
    assert rec.nontrivial_count == 0
    assert rec.soc == sum(solve_unconstrained(s)[0] for s in schedules)
    assert jp is not None


def test_unequal_length_schedules_are_rejected() -> None:
    """Assumption (a): a common makespan is a precondition of the driver.

    reach.py does not extend agent endpoints past |M^i|, so unequal-length
    rows would silently break the decomposition; run_config must fail fast.
    """
    schedules = (
        ("a", "b", "c"),
        ("d", "e"),  # shorter row: violates the common-makespan assumption
    )
    with pytest.raises(AssertionError, match="common makespan"):
        _run("mapfc+ccbs", schedules=schedules)


def test_joint_solver_none_records_solver_none() -> None:
    """A custom configuration whose joint solver returns None records failure."""

    def _never_solves(_schedules: Sequence[Sequence[Any]], **_: Any) -> JointPlan | None:
        return None

    cfg = Configuration(
        name="never_solves",
        kind="mapfc+X",
        joint_solver=_never_solves,
    )
    rec, jp = run_config(
        cfg,
        dataset="test",
        scenario="running",
        num_agents=len(RUNNING_SCHEDULES),
        schedules=RUNNING_SCHEDULES,
    )
    assert rec.soc is None
    assert rec.isr is False
    assert rec.failure_reason == "solver_none"
    assert jp is None


def test_record_provenance_fields_populated() -> None:
    rec, _ = _run("indep_lb")
    assert rec.dataset == "test"
    assert rec.scenario == "running"
    assert rec.num_agents == 3
    assert rec.instance_id == "test/running/n3"
    assert rec.config == "indep_lb"


def test_hybrid_uses_ccbs_when_it_succeeds(monkeypatch) -> None:
    """Hybrid CCBS+JL: when CCBS returns a plan, JL is never called."""
    from mapfc.eval import configs as cfg_mod
    from mapfc.eval import driver as driver_mod

    jl_calls = {"count": 0}

    def _fake_jl(*_args: object, **_kwargs: object) -> JointPlan:
        jl_calls["count"] += 1
        raise AssertionError("Judgelight must not be called when CCBS succeeds")

    class _NeverSession:
        def __enter__(self) -> _NeverSession:
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def solve(self, *_args: object, **_kwargs: object) -> JointPlan:
            return _fake_jl()

    monkeypatch.setattr(cfg_mod, "solve_judgelight", _fake_jl)
    monkeypatch.setattr(driver_mod, "JudgelightSession", _NeverSession)
    rec, _ = _run("mapfc+ccbs+jl")
    assert rec.soc == 4
    assert rec.failure_reason == ""
    assert jl_calls["count"] == 0


def test_hybrid_falls_back_to_jl_when_ccbs_returns_none(monkeypatch) -> None:
    """Hybrid CCBS+JL: when CCBS returns None, JL is invoked with the remaining budget."""
    from mapfc.eval import configs as cfg_mod
    from mapfc.eval import driver as driver_mod
    from mapfc.joint.base import JointPlan as JP

    def _fake_ccbs(schedules: Sequence[Sequence[Any]], **_kwargs: object) -> JointPlan | None:
        return None

    jl_calls = {"count": 0, "budget": 0.0}

    def _fake_jl_solve(
        schedules: Sequence[Sequence[Any]], *, time_limit_sec: float, **_kwargs: object
    ) -> JointPlan:
        jl_calls["count"] += 1
        jl_calls["budget"] = time_limit_sec
        return JP(
            plans=tuple(tuple(s) for s in schedules),
            cost=99,
            solver="fake_jl",
            runtime_s=0.001,
        )

    class _FakeSession:
        def __enter__(self) -> _FakeSession:
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def solve(self, schedules: Sequence[Sequence[Any]], **kwargs: object) -> JointPlan:
            return _fake_jl_solve(schedules, **kwargs)

    monkeypatch.setattr(cfg_mod, "solve_ccbs", _fake_ccbs)
    monkeypatch.setattr(cfg_mod, "solve_judgelight", _fake_jl_solve)
    monkeypatch.setattr(driver_mod, "JudgelightSession", _FakeSession)
    _run("mapfc+ccbs+jl", budget_s=5.0)
    assert jl_calls["count"] >= 1, "JL must be called when CCBS fails"
    assert jl_calls["budget"] > 0.0


def test_hybrid_returns_none_when_no_budget_for_jl(monkeypatch) -> None:
    """Hybrid CCBS+JL: too little remaining budget aborts before calling JL."""
    from mapfc.eval import configs as cfg_mod
    from mapfc.eval import driver as driver_mod

    def _slow_ccbs(
        schedules: Sequence[Sequence[Any]], *, time_limit_sec: float, **_kwargs: object
    ) -> JointPlan | None:
        # Burn the entire wrapper budget so JL has no time left.
        import time as _t

        _t.sleep(min(time_limit_sec, 0.4))
        return None

    jl_calls = {"count": 0}

    def _never_jl(*_args: object, **_kwargs: object) -> JointPlan:
        jl_calls["count"] += 1
        raise AssertionError("JL must not be called when remaining budget is too small")

    class _NeverSession:
        def __enter__(self) -> _NeverSession:
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def solve(self, *_args: object, **_kwargs: object) -> JointPlan:
            return _never_jl()

    monkeypatch.setattr(cfg_mod, "solve_ccbs", _slow_ccbs)
    monkeypatch.setattr(cfg_mod, "solve_judgelight", _never_jl)
    monkeypatch.setattr(driver_mod, "JudgelightSession", _NeverSession)

    # Custom config that gives CCBS the entire 0.3 s budget so nothing's left for JL.
    from mapfc.eval import Configuration

    cfg = Configuration(
        name="ccbs_then_jl_tight",
        kind="mapfc+X",
        joint_solver=cfg_mod._wrap_ccbs_then_judgelight,
        solver_kwargs=(("ccbs_max_sec", 0.4),),
    )
    from mapfc.eval import run_config

    rec, _ = run_config(
        cfg,
        dataset="test",
        scenario="tight",
        num_agents=len(RUNNING_SCHEDULES),
        schedules=RUNNING_SCHEDULES,
        budget_s=0.3,
    )
    assert jl_calls["count"] == 0
    assert rec.soc is None
    assert rec.failure_reason == "solver_none"
