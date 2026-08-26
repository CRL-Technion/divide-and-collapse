"""Python wrapper for Tang et al.'s Judgelight ILP solver.

Serialises a sub-instance to JSON, dispatches to
``scripts/judgelight_solve.py`` in the ``.venv-judgelight`` environment,
and parses the response into a :class:`mapfc.joint.base.JointPlan`.
Used as a baseline alongside our own :func:`mapfc.joint.ccbs.solve_ccbs`.

Two entry points:

- :func:`solve_judgelight` — one-shot dispatch. Opens a fresh subprocess
  per call. Paid ~500 ms of Python + Gurobi initialisation per call.
- :class:`JudgelightSession` — long-lived worker. Pays the ~500 ms boot
  once per session and then dispatches an arbitrary number of solves
  over its lifetime. Use this when running \\Judgelight\\ on several
  sub-instances of one parent instance (the ``mapfc+judgelight`` and
  hybrid configurations of the Q4 driver).

Judgelight requires every agent's trajectory to be the same length;
both entry points pad shorter schedules with trailing waits at the
agent's last vertex, and unpad the response before returning a
:class:`JointPlan` so callers see schedules at their original lengths.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from mapfc.joint.base import JointPlan

REPO_ROOT = Path(__file__).resolve().parents[4]
SUBPROCESS_SCRIPT = REPO_ROOT / "scripts" / "judgelight_solve.py"
DEFAULT_VENV_PY = REPO_ROOT / ".venv-judgelight" / "Scripts" / "python.exe"
DEFAULT_TIME_LIMIT_S = 30.0
DEFAULT_THREADS = 2

V = TypeVar("V", bound=Hashable)


def _to_jsonable(v: object) -> object:
    if isinstance(v, tuple):
        return list(v)
    return v


def _restore(jsonable: object, was_tuple: bool) -> object:
    if was_tuple and isinstance(jsonable, list):
        return tuple(jsonable)
    return jsonable


def _pad_schedules(
    schedules: Sequence[Sequence[V]],
) -> tuple[list[list[V]], int, bool] | None:
    """Pad schedules to a common horizon and return the metadata Judgelight needs.

    Returns ``(padded, horizon, was_tuple)`` or ``None`` if there is nothing
    to send (empty input or zero horizon)."""
    if not schedules:
        return None
    horizon = max(len(s) for s in schedules)
    if horizon == 0:
        return None
    padded: list[list[V]] = []
    for sched in schedules:
        if len(sched) == 0:
            raise ValueError(
                "judgelight subprocess does not accept empty agent schedules; "
                "ensure every agent has at least one vertex"
            )
        agent: list[V] = list(sched)
        last_vertex = agent[-1]
        while len(agent) < horizon:
            agent.append(last_vertex)
        padded.append(agent)
    sample_vertex = padded[0][0]
    was_tuple = isinstance(sample_vertex, tuple)
    return padded, horizon, was_tuple


def _unpack_response(
    response: dict,
    schedules: Sequence[Sequence[V]],
    was_tuple: bool,
    elapsed: float,
) -> JointPlan:
    M_opt_raw = response["M_opt"]
    restored_plans: list[tuple[V, ...]] = []
    for agent_idx, agent in enumerate(M_opt_raw):
        original_len = len(schedules[agent_idx])
        trimmed = agent[:original_len]
        restored = tuple(_restore(v, was_tuple) for v in trimmed)
        restored_plans.append(restored)
    return JointPlan(
        plans=tuple(restored_plans),
        cost=int(response["best_min_cost"]),
        solver="judgelight",
        runtime_s=float(response.get("solve_runtime_s", elapsed)),
        build_runtime_s=float(response.get("build_runtime_s", 0.0)),
    )


class JudgelightSession:
    """Long-lived Judgelight worker.

    Opens a single subprocess running ``scripts/judgelight_solve.py`` in
    session mode (line-delimited JSON I/O). Subsequent :meth:`solve`
    calls reuse the same Python interpreter + warmed-up Gurobi runtime,
    so only the first call pays the ~500 ms boot cost.

    Usage::

        with JudgelightSession() as sess:
            for component in non_trivial_components:
                plan = sess.solve(component.schedules, time_limit_sec=5.0)
                ...

    The session is safe to reuse across many calls. On
    :meth:`close` (or ``__exit__``) the worker is asked to shut down
    cleanly; if it has already died, the cleanup is a no-op.
    """

    def __init__(
        self,
        *,
        venv_python: Path = DEFAULT_VENV_PY,
        script: Path = SUBPROCESS_SCRIPT,
    ) -> None:
        # Lazy: defer the subprocess Popen (and the ~500 ms Python + Gurobi
        # boot it costs) until the first `.solve()` call. This is critical
        # for the hybrid `mapfc+ccbs+jl` configuration, which opens a
        # session per instance but in many instances never falls through
        # to Judgelight (CCBS solves every non-trivial component). With
        # eager Popen, those instances pay the boot cost for nothing.
        self._venv_python = venv_python
        self._script = script
        self._proc: subprocess.Popen[str] | None = None
        self._dead = False
        self._stderr_tail: list[str] = []

    def _ensure_proc(self) -> bool:
        """Spawn the worker lazily on first need. Returns False on failure."""
        if self._proc is not None:
            return True
        if self._dead:
            return False
        env = {**os.environ, "PYTHONUTF8": "1"}
        try:
            self._proc = subprocess.Popen(
                [str(self._venv_python), str(self._script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1,  # line-buffered
            )
        except OSError as exc:
            sys.stderr.write(f"[judgelight_wrapper] failed to spawn worker: {exc}\n")
            self._dead = True
            return False
        return True

    # ------------------------------------------------------------ public API
    def solve(
        self,
        schedules: Sequence[Sequence[V]],
        *,
        time_limit_sec: float = DEFAULT_TIME_LIMIT_S,
        threads: int = DEFAULT_THREADS,
    ) -> JointPlan | None:
        """Solve one sub-instance and return its :class:`JointPlan`, or ``None``.

        Returns ``None`` if (a) the input is empty / zero-horizon and the
        trivial plan is returned by short-circuit; (b) the worker
        reports an error response; (c) the worker has died (e.g.\\ Gurobi
        license expiry, broken pipe). The session is marked dead after
        case (c) and subsequent :meth:`solve` calls also return ``None``.
        """
        if self._dead:
            return None
        meta = _pad_schedules(schedules)
        if meta is None:
            if not schedules:
                return JointPlan(plans=(), cost=0, solver="judgelight", runtime_s=0.0)
            return JointPlan(
                plans=tuple(() for _ in schedules),
                cost=0,
                solver="judgelight",
                runtime_s=0.0,
            )
        if not self._ensure_proc():
            return None
        assert self._proc is not None
        padded, _horizon, was_tuple = meta
        payload: dict[str, Any] = {
            "op": "solve",
            "M": [[_to_jsonable(v) for v in agent] for agent in padded],
            "time_limit_sec": time_limit_sec,
            "threads": threads,
        }
        request = json.dumps(payload) + "\n"
        t_start = time.perf_counter()
        try:
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None
            self._proc.stdin.write(request)
            self._proc.stdin.flush()
            response_line = self._proc.stdout.readline()
        except (BrokenPipeError, ValueError):
            self._mark_dead("write/read on broken pipe")
            return None
        elapsed = time.perf_counter() - t_start

        if not response_line:
            self._mark_dead("worker EOF before response")
            return None

        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as exc:
            sys.stderr.write(
                "[judgelight_wrapper] malformed JSON from worker: "
                f"{exc}; line tail: {response_line[-200:]}\n"
            )
            self._mark_dead("malformed JSON from worker")
            return None

        if not response.get("ok", False):
            sys.stderr.write(
                "[judgelight_wrapper] worker reported error: "
                f"{response.get('error', '<no error message>')}\n"
            )
            return None

        return _unpack_response(response, schedules, was_tuple, elapsed)

    def close(self) -> None:
        # Lazy-spawn case: nothing was ever started.
        if self._proc is None:
            return
        if self._dead or self._proc.poll() is not None:
            self._cleanup()
            return
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
            self._proc.stdin.flush()
            self._proc.stdin.close()
        except (BrokenPipeError, ValueError, OSError):
            pass
        self._cleanup()

    def __enter__(self) -> JudgelightSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -------------------------------------------------------------- internals
    def _mark_dead(self, reason: str) -> None:
        self._dead = True
        if self._proc is None:
            return
        if self._proc.poll() is None:
            with contextlib.suppress(OSError):
                self._proc.kill()
        # Drain stderr for a useful failure trail.
        try:
            if self._proc.stderr is not None:
                tail = self._proc.stderr.read()
                if tail:
                    sys.stderr.write(
                        f"[judgelight_wrapper] worker died ({reason}); "
                        f"stderr tail:\n{tail[-2000:]}\n"
                    )
        except (OSError, ValueError):
            pass

    def _cleanup(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=2.0)
        for handle in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            try:
                if handle is not None:
                    handle.close()
            except (OSError, ValueError):
                pass


def solve_judgelight(
    schedules: Sequence[Sequence[V]],
    *,
    time_limit_sec: float = DEFAULT_TIME_LIMIT_S,
    threads: int = DEFAULT_THREADS,
    venv_python: Path = DEFAULT_VENV_PY,
    script: Path = SUBPROCESS_SCRIPT,
) -> JointPlan | None:
    """One-shot dispatch — open a worker, solve once, shut down.

    Convenience wrapper around :class:`JudgelightSession` for callers
    that issue a single solve. Pays the full ~500 ms boot cost each
    invocation; prefer :class:`JudgelightSession` when batching several
    solves over one parent-instance run.
    """
    with JudgelightSession(venv_python=venv_python, script=script) as sess:
        return sess.solve(
            schedules,
            time_limit_sec=time_limit_sec,
            threads=threads,
        )
