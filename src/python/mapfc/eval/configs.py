"""End-to-end configurations compared in Q4 (experiments.tex).

Each :class:`Configuration` names a dispatch strategy and a joint-solver
factory; the driver consumes them uniformly via :func:`mapfc.eval.driver.run_config`.

The six default configurations:

- ``nocollapse``: return the input schedule as-is; the cost is the move
  count of ``M``. Trivial UB on the joint optimum.
- ``indep_lb``: per-agent ``solve_unconstrained``; the SoC is the sum of
  the per-agent optima ignoring inter-agent collisions. Trivial LB.
- ``judgelight``: monolithic Tang et al. ILP on the full instance.
- ``mapfc+judgelight``: decompose, dispatch singletons via the smart
  sweep, send every non-trivial component to Judgelight.
- ``mapfc+ccbs``: decompose; non-trivial components to the adaptive
  cardinal-first CCBS (``cardinal_trigger_pops=0``: classification from the root).
- ``mapfc+ccbs+jl``: hybrid — CCBS gets the first ``ccbs_max_sec``
  seconds of each component's budget; Judgelight runs on the remainder
  if CCBS returns ``None``. Default ``ccbs_max_sec=1.5`` on a 5 s
  per-instance budget. Combines CCBS's speed on easy/medium components
  with JL's robustness on the warehouse / movingai residue.

"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from mapfc.joint.base import JointPlan
from mapfc.joint.ccbs import solve_ccbs
from mapfc.joint.judgelight_wrapper import JudgelightSession, solve_judgelight


@dataclass(frozen=True)
class Configuration:
    """One configuration compared in Q4.

    ``kind`` selects the dispatch strategy; ``joint_solver`` is the
    callable used for non-trivial components (``mapfc+X``) or for the
    whole instance (``judgelight`` monolithic). ``solver_kwargs`` is
    passed verbatim to the joint solver alongside ``time_limit_sec``.

    ``uses_judgelight`` is a hint to the driver: when ``True``, the
    driver opens a single :class:`JudgelightSession` for the duration
    of the run and passes it through to ``joint_solver`` as the
    ``_session`` keyword argument. This amortises the ~500 ms
    subprocess + Gurobi boot across all per-component calls within one
    instance (relevant for ``judgelight``, ``mapfc+judgelight``, and
    the hybrid ``mapfc+ccbs+jl`` configurations).
    """

    name: str
    kind: str  # "nocollapse" | "indep_lb" | "judgelight" | "mapfc+X"
    joint_solver: Callable[..., JointPlan | None] | None = None
    solver_kwargs: tuple[tuple[str, Any], ...] = ()
    uses_judgelight: bool = False

    def kwargs(self) -> dict[str, Any]:
        return dict(self.solver_kwargs)


def _wrap_judgelight(
    schedules: Sequence[Sequence[Any]],
    *,
    _session: JudgelightSession | None = None,
    **kwargs: Any,
) -> JointPlan | None:
    """Adapter so ``solve_judgelight`` matches the
    ``(schedules, *, time_limit_sec, ...)`` shape.

    When a :class:`JudgelightSession` is supplied via the keyword-only
    ``_session`` parameter, the call is routed through it (no fresh
    subprocess boot); otherwise the legacy one-shot
    :func:`solve_judgelight` is used.
    """
    if _session is not None:
        return _session.solve(schedules, **kwargs)
    return solve_judgelight(schedules, **kwargs)


_JL_SUBPROCESS_OVERHEAD_S = 0.2


def _wrap_ccbs_then_judgelight(
    schedules: Sequence[Sequence[Any]],
    *,
    time_limit_sec: float,
    _session: JudgelightSession | None = None,
    **kwargs: Any,
) -> JointPlan | None:
    """Hybrid: CCBS first, Judgelight fallback on the remaining budget.

    CCBS is given the lesser of the per-component budget and
    ``ccbs_max_sec`` (default ``1.5`` s). On success its :class:`JointPlan`
    is returned immediately. On failure (``None`` — typically a
    ``time_limit_sec`` exhaustion or an infeasible search) Judgelight is
    invoked on the remaining wall-clock minus a small subprocess-startup
    margin. If too little budget remains for JL to make progress the
    caller sees ``None``.

    When a :class:`JudgelightSession` is supplied via the keyword-only
    ``_session`` parameter, the JL fallback is routed through the warm
    session and the subprocess-startup margin shrinks to a few
    milliseconds (just the JSON write/read), so the budget guard is
    almost always satisfied.

    Tagged-kwarg conventions match the underlying solvers:

    - CCBS keys: ``cardinal_trigger_pops``, ``use_disjoint_split``.
    - Judgelight keys: ``threads``.
    - Hybrid-only key: ``ccbs_max_sec``.

    Any unrecognised kwargs are silently dropped so the same
    ``solver_kwargs`` tuple can be passed to both single-solver and
    hybrid configurations.
    """
    t0 = time.perf_counter()
    ccbs_max_sec = float(kwargs.get("ccbs_max_sec", 1.5))
    threads = int(kwargs.get("threads", 1))
    cardinal_trigger_pops = int(kwargs.get("cardinal_trigger_pops", 0))
    use_disjoint_split = bool(kwargs.get("use_disjoint_split", True))

    ccbs_budget = min(time_limit_sec, ccbs_max_sec)
    if ccbs_budget > 0:
        ccbs_result = solve_ccbs(
            schedules,
            time_limit_sec=ccbs_budget,
            cardinal_trigger_pops=cardinal_trigger_pops,
            use_disjoint_split=use_disjoint_split,
        )
        if ccbs_result is not None:
            return ccbs_result

    elapsed = time.perf_counter() - t0
    remaining = time_limit_sec - elapsed
    # With a warm session, the fallback overhead is just IPC; without one,
    # the legacy ~200 ms subprocess boot margin still applies.
    margin = 0.01 if _session is not None else _JL_SUBPROCESS_OVERHEAD_S
    if remaining < margin:
        return None
    if _session is not None:
        return _session.solve(schedules, time_limit_sec=remaining, threads=threads)
    return solve_judgelight(
        schedules,
        time_limit_sec=remaining,
        threads=threads,
    )


def _wrap_size_gated_hybrid(
    schedules: Sequence[Sequence[Any]],
    *,
    time_limit_sec: float,
    _session: JudgelightSession | None = None,
    **kwargs: Any,
) -> JointPlan | None:
    """Size-gated hybrid: route large components straight to Judgelight.

    Components with more than ``gate_tau`` agents (default ``30``, read off
    the Q3 phase transition where C-CBS stops beating Judgelight) skip the
    C-CBS attempt entirely and go straight to the (warm) Judgelight session,
    so a large core never burns the ``ccbs_max_sec`` budget before falling
    back. Components of size ``<= gate_tau`` are handled exactly as the plain
    hybrid :func:`_wrap_ccbs_then_judgelight` (C-CBS first, JL on the
    remaining budget).

    The gate only reallocates *which* solver sees a component; every returned
    plan is still either a C-CBS optimum or a Judgelight solution, so the
    soundness/optimality guarantees of the two underlying solvers are
    unchanged. ``gate_tau`` is read from ``kwargs`` (default ``30``); all
    other kwargs are forwarded to the underlying hybrid/JL calls.
    """
    gate_tau = int(kwargs.get("gate_tau", 30))
    if len(schedules) <= gate_tau:
        return _wrap_ccbs_then_judgelight(
            schedules, time_limit_sec=time_limit_sec, _session=_session, **kwargs
        )
    threads = int(kwargs.get("threads", 1))
    if _session is not None:
        return _session.solve(schedules, time_limit_sec=time_limit_sec, threads=threads)
    return solve_judgelight(schedules, time_limit_sec=time_limit_sec, threads=threads)


CONFIGS: tuple[Configuration, ...] = (
    Configuration(name="nocollapse", kind="nocollapse"),
    Configuration(name="indep_lb", kind="indep_lb"),
    Configuration(
        name="judgelight",
        kind="judgelight",
        joint_solver=_wrap_judgelight,
        solver_kwargs=(("threads", 1),),
        uses_judgelight=True,
    ),
    Configuration(
        name="mapfc+judgelight",
        kind="mapfc+X",
        joint_solver=_wrap_judgelight,
        solver_kwargs=(("threads", 1),),
        uses_judgelight=True,
    ),
    Configuration(
        name="mapfc+ccbs",
        kind="mapfc+X",
        joint_solver=solve_ccbs,
        solver_kwargs=(("cardinal_trigger_pops", 0),),
    ),
    Configuration(
        name="mapfc+ccbs+jl",
        kind="mapfc+X",
        joint_solver=_wrap_ccbs_then_judgelight,
        solver_kwargs=(
            ("ccbs_max_sec", 1.5),
            ("cardinal_trigger_pops", 0),
            ("threads", 1),
        ),
        uses_judgelight=True,
    ),
    Configuration(
        name="mapfc+ccbs+jl+gate20",
        kind="mapfc+X",
        joint_solver=_wrap_size_gated_hybrid,
        solver_kwargs=(
            ("gate_tau", 20),
            ("ccbs_max_sec", 1.5),
            ("cardinal_trigger_pops", 0),
            ("threads", 1),
        ),
        uses_judgelight=True,
    ),
    Configuration(
        name="mapfc+ccbs+jl+gate30",
        kind="mapfc+X",
        joint_solver=_wrap_size_gated_hybrid,
        solver_kwargs=(
            ("gate_tau", 30),
            ("ccbs_max_sec", 1.5),
            ("cardinal_trigger_pops", 0),
            ("threads", 1),
        ),
        uses_judgelight=True,
    ),
    Configuration(
        name="mapfc+ccbs+jl+gate40",
        kind="mapfc+X",
        joint_solver=_wrap_size_gated_hybrid,
        solver_kwargs=(
            ("gate_tau", 40),
            ("ccbs_max_sec", 1.5),
            ("cardinal_trigger_pops", 0),
            ("threads", 1),
        ),
        uses_judgelight=True,
    ),
)

CONFIG_BY_NAME: dict[str, Configuration] = {c.name: c for c in CONFIGS}
ALL_CONFIG_NAMES: tuple[str, ...] = tuple(c.name for c in CONFIGS)

# Size-gated hybrids are excluded from the default sweep; select one
# explicitly with `--configs` (the paper's recommended configuration is
# `mapfc+ccbs+jl+gate40`).
DEFAULT_CONFIG_NAMES: tuple[str, ...] = tuple(c.name for c in CONFIGS if "gate" not in c.name)
