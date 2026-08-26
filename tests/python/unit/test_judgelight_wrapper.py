"""Smoke test for the Judgelight subprocess wrapper.

Skipped when ``.venv-judgelight`` is not present (CI does not install Tang
et al.'s pinned dependencies); the local developer setup, which already
runs ``scripts/setup_judgelight_venv.sh`` for the Q1/Q2 dumpers, does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mapfc.joint import baseline_cost
from mapfc.joint.judgelight_wrapper import DEFAULT_VENV_PY, solve_judgelight

pytestmark = pytest.mark.skipif(
    not DEFAULT_VENV_PY.exists(),
    reason=".venv-judgelight not present; run scripts/setup_judgelight_venv.sh",
)


def test_judgelight_running_example_single_agent() -> None:
    """Agent 1 of the running example: M=<a,b,c,b,e> -> cost 2 plan <a,b,b,b,e>."""
    schedules = [("a", "b", "c", "b", "e")]
    result = solve_judgelight(schedules, time_limit_sec=10.0, threads=2)
    assert result is not None
    assert result.cost == 2
    assert len(result.plans) == 1
    assert baseline_cost(result.plans) == result.cost
    assert result.plans[0][0] == "a"
    assert result.plans[0][-1] == "e"


def test_judgelight_two_agent_no_conflict() -> None:
    schedules = [("a", "b", "c"), ("d", "e", "f")]
    result = solve_judgelight(schedules, time_limit_sec=10.0, threads=2)
    assert result is not None
    assert result.cost == baseline_cost(schedules)


def test_judgelight_grid_cell_vertex() -> None:
    schedules = [((0, 0), (0, 1), (0, 0), (1, 0))]
    result = solve_judgelight(schedules, time_limit_sec=10.0, threads=2)
    assert result is not None
    assert result.cost == 1
    assert result.plans[0][0] == (0, 0)
    assert result.plans[0][-1] == (1, 0)
    for v in result.plans[0]:
        assert isinstance(v, tuple)


def test_judgelight_empty_schedules() -> None:
    result = solve_judgelight([])
    assert result is not None
    assert result.cost == 0
    assert result.plans == ()


def test_judgelight_padding_reversed_on_return() -> None:
    """Schedules of different lengths get padded for the subprocess; the
    returned plans must come back at their original lengths."""
    schedules = [("a", "b", "c", "b", "e"), ("d", "d")]
    result = solve_judgelight(schedules, time_limit_sec=10.0, threads=2)
    assert result is not None
    assert len(result.plans[0]) == 5
    assert len(result.plans[1]) == 2


def test_default_venv_path_resolves(tmp_path: Path) -> None:
    """Sanity check on the venv-python path resolver; if this fails the
    REPO_ROOT computation in the wrapper has drifted from the layout."""
    assert DEFAULT_VENV_PY.parent.name == "Scripts"
    assert DEFAULT_VENV_PY.name == "python.exe"
