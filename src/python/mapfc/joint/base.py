"""Joint-solver interface: the contract every MAPFC joint solver implements.

A joint solver takes a sub-instance of MAPFC (a tuple of single-agent
schedules) and returns either a ``JointPlan`` (a tuple of per-agent
collapse-derived trajectories with their summed cost) or ``None`` when the
solver cannot find a feasible plan within its budget.

Both the decomposition pipeline of :mod:`mapfc.decompose` and the monolithic
end-to-end driver call into one of three solvers conforming to this
interface: :func:`mapfc.joint.judgelight_wrapper.solve_judgelight`, the
exact CBS-style :func:`mapfc.joint.ccbs.solve_ccbs`, and the
and the Judgelight wrapper.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

V = TypeVar("V", bound=Hashable)


@dataclass(frozen=True)
class JointPlan:
    """The return value of a joint MAPFC solver."""

    plans: tuple[tuple[Hashable, ...], ...]
    """One trajectory per agent. Each entry ``plans[i]`` is the post-collapse
    sequence of vertex positions for agent ``i``."""

    cost: int
    """Total move count summed across agents, ``sum_i c(plans[i])``."""

    solver: str
    """Identifier of the solver that produced this plan, e.g. ``"judgelight"``,
    ``"ccbs"``, ``"judgelight"``."""

    runtime_s: float
    """Wall-clock from solver entry to plan returned. The convention matches
    Tang et al.'s ``ilp_time`` for ``judgelight``; for our CBS solver it
    is the full search wall-clock."""

    build_runtime_s: float = 0.0
    """For ILP-style solvers, the wall-clock of model construction and load
    that is *not* counted in ``runtime_s``. For our CBS solver this is
    always zero. The split exists so ``experiments.tex``'s two wall-clock
    columns (total vs solver-only) can be filled mechanically."""


class JointSolver(Protocol):
    """Structural type that every joint solver implements."""

    def __call__(self, schedules: Sequence[Sequence[V]], **kwargs: object) -> JointPlan | None: ...


def baseline_cost(plans: Sequence[Sequence[V]]) -> int:
    """Aggregate cost: sum over agents of the per-trajectory move count."""
    total = 0
    for plan in plans:
        for t in range(len(plan) - 1):
            if plan[t] != plan[t + 1]:
                total += 1
    return total
