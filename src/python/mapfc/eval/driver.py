"""End-to-end driver for one ``(instance, configuration)`` pair.

:func:`run_config` consumes a :class:`Configuration` and a sequence of
per-agent input schedules ``M = (M^1, ..., M^N)`` (an instance of
MAPF-Collapse) and produces:

- a :class:`MetricRecord` — one parquet row's worth of headline numbers
  (SoC, saving ratio, ISR, the two wall-clock columns required by
  ``experiments.tex``, plus decomposition diagnostics);
- the materialised :class:`JointPlan` for downstream feasibility checks
  (None on failure).

The driver never raises; failures are recorded in
``MetricRecord.failure_reason`` and emitted as ``soc=None``,
``isr=False``.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from mapfc.decompose import DecompositionResult, decompose
from mapfc.eval.configs import Configuration
from mapfc.joint.base import JointPlan, baseline_cost
from mapfc.joint.judgelight_wrapper import JudgelightSession
from mapfc.single_agent.collapse_dag import solve_unconstrained

V = TypeVar("V", bound=Hashable)


@dataclass(frozen=True)
class MetricRecord:
    """One row of Q4 results.

    Field ordering mirrors the design doc:
    provenance, outcome, wall-clock split, decomposition diagnostics,
    bounds, failure provenance.
    """

    dataset: str
    scenario: str
    num_agents: int
    instance_id: str
    config: str

    soc: int | None
    input_soc: int
    saved_soc: int | None
    saving_ratio: float | None
    isr: bool

    runtime_total_s: float
    runtime_solver_only_s: float
    build_runtime_s: float
    decomp_runtime_s: float
    singleton_dispatch_s: float
    joint_solver_s: float

    singleton_count: int
    nontrivial_count: int
    max_nontrivial_size: int
    singleton_fraction: float
    edges_count: int
    reach_total_cells: int

    decomp_lb: int | None
    decomp_ub: int

    failure_reason: str


@dataclass
class _State:
    """Mutable accumulator used while a single run is in progress."""

    cfg: Configuration
    dataset: str
    scenario: str
    num_agents: int
    instance_id: str
    input_soc: int

    t_entry: float = 0.0
    soc: int | None = None
    decomp_runtime_s: float = 0.0
    singleton_dispatch_s: float = 0.0
    joint_solver_s: float = 0.0
    build_runtime_s: float = 0.0

    singleton_count: int = 0
    nontrivial_count: int = 0
    max_nontrivial_size: int = 0
    singleton_fraction: float = 0.0
    edges_count: int = 0
    reach_total_cells: int = 0
    decomp_lb: int | None = None

    failure_reason: str = ""
    joint_plan: JointPlan | None = None

    def freeze(self, *, budget_s: float) -> MetricRecord:
        elapsed = time.perf_counter() - self.t_entry
        soc = self.soc
        saved = self.input_soc - soc if soc is not None else None
        ratio = (
            (saved / self.input_soc)
            if (soc is not None and self.input_soc > 0)
            else (None if soc is None else 0.0)
        )
        isr = soc is not None and elapsed <= budget_s
        kind = self.cfg.kind
        if kind in ("nocollapse", "indep_lb"):
            solver_only = self.singleton_dispatch_s if kind == "indep_lb" else 0.0
        elif kind == "judgelight":
            solver_only = self.joint_solver_s
        else:  # mapfc+X
            solver_only = self.singleton_dispatch_s + self.joint_solver_s
        return MetricRecord(
            dataset=self.dataset,
            scenario=self.scenario,
            num_agents=self.num_agents,
            instance_id=self.instance_id,
            config=self.cfg.name,
            soc=soc,
            input_soc=self.input_soc,
            saved_soc=saved,
            saving_ratio=ratio,
            isr=isr,
            runtime_total_s=elapsed,
            runtime_solver_only_s=solver_only,
            build_runtime_s=self.build_runtime_s,
            decomp_runtime_s=self.decomp_runtime_s,
            singleton_dispatch_s=self.singleton_dispatch_s,
            joint_solver_s=self.joint_solver_s,
            singleton_count=self.singleton_count,
            nontrivial_count=self.nontrivial_count,
            max_nontrivial_size=self.max_nontrivial_size,
            singleton_fraction=self.singleton_fraction,
            edges_count=self.edges_count,
            reach_total_cells=self.reach_total_cells,
            decomp_lb=self.decomp_lb,
            decomp_ub=self.input_soc,
            failure_reason=self.failure_reason,
        )


def _record_decomp(state: _State, decomp: DecompositionResult) -> None:
    state.decomp_runtime_s = decomp.h_construction_time_s
    state.singleton_count = decomp.n_singletons
    state.nontrivial_count = len(decomp.non_trivial_components)
    state.max_nontrivial_size = decomp.max_non_trivial_size
    state.singleton_fraction = decomp.singleton_fraction
    state.edges_count = decomp.edges_count
    state.reach_total_cells = decomp.reach_total_cells


def _run_nocollapse(schedules: Sequence[Sequence[V]], state: _State) -> None:
    state.soc = state.input_soc
    state.joint_plan = JointPlan(
        plans=tuple(tuple(s) for s in schedules),
        cost=state.input_soc,
        solver="nocollapse",
        runtime_s=0.0,
        build_runtime_s=0.0,
    )


def _run_indep_lb(schedules: Sequence[Sequence[V]], state: _State) -> None:
    total = 0
    plans: list[tuple[V, ...]] = []
    for sched in schedules:
        t0 = time.perf_counter()
        cost, plan = solve_unconstrained(sched)
        state.singleton_dispatch_s += time.perf_counter() - t0
        total += cost
        plans.append(tuple(plan))
    state.soc = total
    state.joint_plan = JointPlan(
        plans=tuple(plans),
        cost=total,
        solver="indep_lb",
        runtime_s=state.singleton_dispatch_s,
        build_runtime_s=0.0,
    )


def _judgelight_session_cm(cfg: Configuration):
    """Open a :class:`JudgelightSession` for the duration of one instance
    if the configuration needs one; otherwise yield ``None``.

    The session is created BEFORE the timing window is entered by the
    driver, so its subprocess boot cost (~500 ms) shows up inside the
    first ``cfg.joint_solver(...)`` call rather than as a separate
    accounting line. That is the desired attribution: it is part of
    the joint-solver time, and amortising it is exactly the win the
    session unlocks.
    """
    if cfg.uses_judgelight:
        return JudgelightSession()
    return contextlib.nullcontext(None)


def _run_judgelight(schedules: Sequence[Sequence[V]], state: _State, *, budget_s: float) -> None:
    assert state.cfg.joint_solver is not None
    with _judgelight_session_cm(state.cfg) as session:
        extra: dict[str, object] = {"_session": session} if session is not None else {}
        t0 = time.perf_counter()
        jp = state.cfg.joint_solver(
            schedules, time_limit_sec=budget_s, **extra, **state.cfg.kwargs()
        )
        state.joint_solver_s += time.perf_counter() - t0
    if jp is None:
        state.failure_reason = "solver_none"
        return
    state.build_runtime_s += jp.build_runtime_s
    state.soc = jp.cost
    state.joint_plan = jp


def _run_mapfc_plus(schedules: Sequence[Sequence[V]], state: _State, *, budget_s: float) -> None:
    assert state.cfg.joint_solver is not None
    try:
        decomp = decompose(schedules)
    except Exception as exc:  # pragma: no cover
        state.failure_reason = f"exception:{type(exc).__name__}"
        return
    _record_decomp(state, decomp)
    n = decomp.n_agents

    per_agent_plans: list[tuple[V, ...] | None] = [None] * n
    component_socs: list[int] = []
    singleton_costs: list[int] = []

    for i in decomp.singletons:
        if time.perf_counter() - state.t_entry > budget_s:
            state.failure_reason = "timeout"
            return
        t0 = time.perf_counter()
        cost, plan = solve_unconstrained(schedules[i])
        state.singleton_dispatch_s += time.perf_counter() - t0
        per_agent_plans[i] = tuple(plan)
        singleton_costs.append(cost)

    with _judgelight_session_cm(state.cfg) as session:
        extra: dict[str, object] = {"_session": session} if session is not None else {}
        for comp in decomp.non_trivial_components:
            ids = tuple(sorted(comp))
            remaining = max(0.0, budget_s - (time.perf_counter() - state.t_entry))
            if remaining <= 0.0:
                state.failure_reason = "timeout"
                return
            sub = [schedules[i] for i in ids]
            t0 = time.perf_counter()
            jp = state.cfg.joint_solver(
                sub, time_limit_sec=remaining, **extra, **state.cfg.kwargs()
            )
            state.joint_solver_s += time.perf_counter() - t0
            if jp is None:
                state.failure_reason = "solver_none"
                return
            state.build_runtime_s += jp.build_runtime_s
            component_socs.append(jp.cost)
            for k, agent_id in enumerate(ids):
                per_agent_plans[agent_id] = jp.plans[k]

    soc = sum(singleton_costs) + sum(component_socs)
    state.soc = soc
    state.decomp_lb = soc

    full_plans = tuple(
        per_agent_plans[i] if per_agent_plans[i] is not None else tuple(schedules[i])
        for i in range(n)
    )
    state.joint_plan = JointPlan(
        plans=full_plans,
        cost=soc,
        solver=state.cfg.name,
        runtime_s=state.singleton_dispatch_s + state.joint_solver_s,
        build_runtime_s=state.build_runtime_s,
    )


def run_config(
    cfg: Configuration,
    *,
    dataset: str,
    scenario: str,
    num_agents: int,
    schedules: Sequence[Sequence[V]],
    budget_s: float = 5.0,
) -> tuple[MetricRecord, JointPlan | None]:
    """Run one ``(instance, configuration)`` pair end-to-end.

    Returns ``(record, joint_plan)``. Once dispatch begins the driver never
    raises; if the configured dispatch fails the record carries ``soc=None``,
    ``isr=False``, and a non-empty ``failure_reason``.

    Precondition: all rows of ``schedules`` share a common makespan
    (assumption (a) of the paper, the standard MAPF-solution form in which every
    agent waits at its goal until the last arrival). The decomposition
    (:mod:`mapfc.decompose.reach`) does not extend agent endpoints past
    ``|M^i|``, so unequal-length rows would make the interaction graph, and
    hence the decomposition, unsound. A violation raises ``AssertionError``.
    """
    lengths = {len(s) for s in schedules}
    assert len(lengths) <= 1, (
        f"MAPF-Collapse assumes a common makespan across input rows; got row "
        f"lengths {sorted(lengths)}. reach.py does not extend endpoints past "
        f"|M^i|, so unequal lengths would make the decomposition unsound."
    )

    instance_id = f"{dataset}/{scenario}/n{num_agents}"
    state = _State(
        cfg=cfg,
        dataset=dataset,
        scenario=scenario,
        num_agents=num_agents,
        instance_id=instance_id,
        input_soc=baseline_cost(schedules),
        t_entry=time.perf_counter(),
    )

    try:
        if cfg.kind == "nocollapse":
            _run_nocollapse(schedules, state)
        elif cfg.kind == "indep_lb":
            _run_indep_lb(schedules, state)
        elif cfg.kind == "judgelight":
            _run_judgelight(schedules, state, budget_s=budget_s)
        elif cfg.kind == "mapfc+X":
            _run_mapfc_plus(schedules, state, budget_s=budget_s)
        else:  # pragma: no cover
            state.failure_reason = f"unknown_kind:{cfg.kind}"
    except Exception as exc:  # pragma: no cover
        state.failure_reason = f"exception:{type(exc).__name__}"

    return state.freeze(budget_s=budget_s), state.joint_plan
