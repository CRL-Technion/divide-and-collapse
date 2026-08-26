"""Hypothesis profile registration for the stress suite.

Two profiles, selected via the ``MAPFC_HYPOTHESIS_PROFILE`` env var:

- ``stress-dev`` (default): 50 examples; suitable for a developer running
  ``pytest -m slow tests/python/stress/`` on demand.
- ``stress-ci``: 500 examples; for the nightly CI job that exhaustively
  fuzzes the cross-oracle invariants.

Deadlines are disabled because every example invokes Judgelight, which
launches a Python subprocess and a Gurobi optimisation — well above
Hypothesis' default 200 ms per-example deadline.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "stress-dev",
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
settings.register_profile(
    "stress-ci",
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
settings.load_profile(os.environ.get("MAPFC_HYPOTHESIS_PROFILE", "stress-dev"))
