"""Brute-force cross-validation for CCBS on a deterministic test suite.

The brute-force reference enumerates every collapse-derived plan per agent
(across all disjoint-or-touching subsets of valid collapse intervals),
takes the cross product, filters by joint vertex-conflict-freeness, and
returns the minimum joint cost. The smart sweep then has to match it.

We avoid Hypothesis here: the small-input brute force is already
exhaustive and a deterministic enumeration over short alphabets and
small schedule lengths covers every interesting structural pattern.
"""

from __future__ import annotations

from itertools import product

import pytest

from mapfc.joint import baseline_cost
from mapfc.joint.ccbs import find_vertex_conflict, solve_ccbs

ALPHABET = ("a", "b", "c")


def _enumerate_plans(schedule: list[str]) -> list[tuple[str, ...]]:
    """Yield every plan obtainable from ``schedule`` by a disjoint-or-touching
    subset of valid collapse intervals."""
    k = len(schedule) - 1
    if k < 0:
        return [()]
    intervals = [
        (a, b) for a in range(k) for b in range(a + 1, k + 1) if schedule[a] == schedule[b]
    ]
    n = len(intervals)
    seen: set[tuple[str, ...]] = set()
    plans: list[tuple[str, ...]] = []
    for mask in range(1 << n):
        chosen = sorted(intervals[i] for i in range(n) if mask & (1 << i))
        if any(chosen[i + 1][0] < chosen[i][1] for i in range(len(chosen) - 1)):
            continue
        plan = list(schedule)
        for a, b in chosen:
            anchor = schedule[a]
            for s in range(a, b + 1):
                plan[s] = anchor
        t = tuple(plan)
        if t in seen:
            continue
        seen.add(t)
        plans.append(t)
    return plans


def _joint_conflict_free(plans: list[tuple[str, ...]]) -> bool:
    horizon = max((len(p) for p in plans), default=0)
    for t in range(horizon):
        occupants: dict[str, int] = {}
        for i, p in enumerate(plans):
            if not p:
                continue
            v = p[t] if t < len(p) else p[-1]
            if v in occupants:
                return False
            occupants[v] = i
    return True


def _brute_joint_cost(schedules: list[list[str]]) -> int | None:
    if not schedules:
        return 0
    per_agent = [_enumerate_plans(s) for s in schedules]
    best: int | None = None
    for combo in product(*per_agent):
        if not _joint_conflict_free(list(combo)):
            continue
        cost = sum(sum(1 for t in range(len(p) - 1) if p[t] != p[t + 1]) for p in combo)
        if best is None or cost < best:
            best = cost
    return best


def _enumerate_pairs(max_len: int) -> list[tuple[list[str], list[str]]]:
    """Every pair of schedules over ALPHABET with both lengths in [1, max_len]."""
    out: list[tuple[list[str], list[str]]] = []
    for la in range(1, max_len + 1):
        for lb in range(1, max_len + 1):
            for sa in product(ALPHABET, repeat=la):
                for sb in product(ALPHABET, repeat=lb):
                    out.append((list(sa), list(sb)))
    return out


def _enumerate_triples(max_len: int) -> list[tuple[list[str], list[str], list[str]]]:
    """Every triple of schedules over ALPHABET with all lengths in [1, max_len]."""
    out: list[tuple[list[str], list[str], list[str]]] = []
    for la in range(1, max_len + 1):
        for lb in range(1, max_len + 1):
            for lc in range(1, max_len + 1):
                for sa in product(ALPHABET, repeat=la):
                    for sb in product(ALPHABET, repeat=lb):
                        for sc in product(ALPHABET, repeat=lc):
                            out.append((list(sa), list(sb), list(sc)))
    return out


PAIRS = _enumerate_pairs(max_len=2)
TRIPLES = _enumerate_triples(max_len=2)


@pytest.mark.parametrize("pair", PAIRS)
def test_two_agent_ccbs_matches_brute_force(pair: tuple[list[str], list[str]]) -> None:
    schedules = list(pair)
    brute = _brute_joint_cost(schedules)
    result = solve_ccbs(schedules)
    if brute is None:
        assert result is None, (schedules, result)
        return
    assert result is not None, (schedules, brute)
    assert result.cost == brute, (schedules, brute, result.cost)


@pytest.mark.parametrize("pair", PAIRS)
def test_two_agent_returned_plan_is_conflict_free(
    pair: tuple[list[str], list[str]],
) -> None:
    result = solve_ccbs(list(pair))
    if result is None:
        return
    assert find_vertex_conflict(result.plans) is None


@pytest.mark.parametrize("pair", PAIRS)
def test_two_agent_cost_matches_baseline_of_plan(
    pair: tuple[list[str], list[str]],
) -> None:
    result = solve_ccbs(list(pair))
    if result is None:
        return
    assert baseline_cost(result.plans) == result.cost


@pytest.mark.parametrize("triple", TRIPLES)
def test_three_agent_ccbs_matches_brute_force(
    triple: tuple[list[str], list[str], list[str]],
) -> None:
    schedules = list(triple)
    brute = _brute_joint_cost(schedules)
    result = solve_ccbs(schedules)
    if brute is None:
        assert result is None, (schedules, result)
        return
    assert result is not None, (schedules, brute)
    assert result.cost == brute, (schedules, brute, result.cost)
