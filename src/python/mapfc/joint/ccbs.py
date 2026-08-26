"""Conflict-Based Search for MAPF-Collapse (Alg. 3 of improvements.tex).

Best-first CBS over per-agent constraint sets. Each high-level node carries a
per-agent ``(negative, positive)`` constraint pair and the per-agent
constrained-optimal plans of the constrained smart sweep (Prop. 4.1). The
lower bound on the node's cost is the sum of the per-agent constrained
optima. On pop we look for a vertex conflict in the joint plan; if none,
the plan is optimal.

Engineering ingredients:

- **Disjoint splitting** (App. A.4 — CBS-PC): on a vertex conflict
  ``(i, j, v, t)`` we pick one agent (here the one with fewer existing
  constraints — ties go to the lower index) and branch ``(i positive,
  i negative)``. The left child adds ``(v, t)`` to agent i's negatives; the
  right child adds ``(v, t)`` to agent i's positives AND to agent j's
  negatives (since if i must be at ``(v, t)``, j cannot). The two branches
  are *disjoint* — every joint plan is in exactly one — so no work is
  duplicated, and positive constraints prune the other agents' search
  spaces transitively. Toggle with ``use_disjoint_split`` (default True).
- **Adaptive A.2 cardinal-conflict prioritisation** (Alg.~\\ref{alg:eng-cardinal}):
  every vertex conflict can be classified by the change in per-agent LB if
  its cell were added as a constraint: cardinal (both LBs strictly increase)
  > semi (one increases) > non. The classifier is *adaptive* — it pays two
  constrained-sweep solves per conflict, which is wasted work on small
  components that resolve in a few pops. The selector starts with the cheap
  earliest-time conflict (Lem.~no-edge-conflicts ordering) and only switches
  to cardinal-first once the high-level pop count exceeds
  ``cardinal_trigger_pops`` (default 0: cardinal-first from the root).
- Per-call cache of constrained-sweep solves keyed by ``(agent_idx,
  negative, positive)`` so the A.2 lookahead and the subsequent branch
  never duplicate work.

Other App. A ingredients (range constraints, rectangle/corridor symmetry,
bypass, mutex propagation, reach-set tightening) remain deferred.
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from mapfc.joint.base import JointPlan
from mapfc.single_agent.collapse_dag import solve_unconstrained
from mapfc.single_agent.constrained_dag import solve_constrained

V = TypeVar("V", bound=Hashable)

_INF = float("inf")


@dataclass(frozen=True)
class Conflict:
    """A vertex conflict in a joint plan: agents ``i < j`` collide at vertex
    ``v`` at time ``t``."""

    i: int
    j: int
    v: Hashable
    t: int


Cell = tuple[Any, int]
NegPos = tuple[frozenset[Cell], frozenset[Cell]]


class _ConflictMap:
    """Incremental tracker of vertex conflicts for a joint plan.

    For each occupied cell ``(v, t)`` stores the set of agents at that cell,
    and tracks the subset of cells with two or more agents (the "conflict
    cells"). The map clones to a child in time proportional to the number
    of occupied cells, and updates for one agent's plan-replacement in
    time proportional to the horizon. This replaces the
    ``O(N x horizon)`` re-scan that :func:`_iter_vertex_conflicts` does
    on every CCBS pop, which dominates the per-pop cost on large
    components (>20 agents).
    """

    __slots__ = ("agents_at", "conflict_cells")

    def __init__(self) -> None:
        # (v, t) -> frozenset of agent indices currently at that cell.
        self.agents_at: dict[tuple[Any, int], frozenset[int]] = {}
        # t -> { v -> frozenset of agents }, restricted to multi-occupant cells.
        self.conflict_cells: dict[int, dict[Any, frozenset[int]]] = {}

    @classmethod
    def from_plans(cls, plans: Sequence[Sequence[Any]]) -> _ConflictMap:
        cm = cls()
        for i, p in enumerate(plans):
            if not p:
                continue
            for t, v in enumerate(p):
                cm._add(i, v, t)
        return cm

    def clone(self) -> _ConflictMap:
        cm = _ConflictMap()
        cm.agents_at = dict(self.agents_at)
        cm.conflict_cells = {t: dict(vmap) for t, vmap in self.conflict_cells.items()}
        return cm

    def replace_agent(
        self,
        idx: int,
        old_plan: Sequence[Any],
        new_plan: Sequence[Any],
    ) -> None:
        # old_plan and new_plan share length (horizon-padded by the caller).
        for t in range(len(old_plan)):
            old_v = old_plan[t]
            new_v = new_plan[t]
            if old_v == new_v:
                continue
            self._remove(idx, old_v, t)
            self._add(idx, new_v, t)

    def has_conflict(self) -> bool:
        return bool(self.conflict_cells)

    def first_conflict(self) -> Conflict | None:
        """Earliest-time conflict; among ties picks the smallest (i, j, v)
        triple. Matches the property "smaller minimum-agent index goes first",
        which preserves the behaviour the existing tests rely on for the
        single-multi-occupant-per-timestep case."""
        if not self.conflict_cells:
            return None
        t_min = min(self.conflict_cells)
        vmap = self.conflict_cells[t_min]
        best: tuple[int, int, Any] | None = None
        for v, agents in vmap.items():
            srt = sorted(agents)
            cand = (srt[0], srt[1], v)
            if best is None or cand < best:
                best = cand
        assert best is not None
        i, j, v = best
        return Conflict(i=i, j=j, v=v, t=t_min)

    def iter_conflicts_sorted(self):  # type: ignore[no-untyped-def]
        """All conflicts in deterministic (t, min-agent, second-agent, v) order.

        Used by the cardinal selector to iterate candidate branch conflicts.
        """
        for t in sorted(self.conflict_cells):
            vmap = self.conflict_cells[t]
            entries: list[tuple[list[int], Any]] = []
            for v, agents in vmap.items():
                entries.append((sorted(agents), v))
            entries.sort(key=lambda e: (e[0][0], e[0][1], e[1]))
            for srt, v in entries:
                for x in range(len(srt)):
                    for y in range(x + 1, len(srt)):
                        yield Conflict(i=srt[x], j=srt[y], v=v, t=t)

    # ------------------------------------------------------------ internals
    def _add(self, idx: int, v: Any, t: int) -> None:
        key = (v, t)
        existing = self.agents_at.get(key)
        if existing is None:
            self.agents_at[key] = frozenset((idx,))
            return
        new_set = existing | {idx}
        self.agents_at[key] = new_set
        if len(new_set) >= 2:
            self.conflict_cells.setdefault(t, {})[v] = new_set

    def _remove(self, idx: int, v: Any, t: int) -> None:
        key = (v, t)
        existing = self.agents_at.get(key)
        if existing is None or idx not in existing:
            return
        new_set = existing - {idx}
        if not new_set:
            del self.agents_at[key]
        else:
            self.agents_at[key] = new_set
        # Maintain conflict_cells consistency.
        vmap = self.conflict_cells.get(t)
        if vmap is None:
            return
        if len(new_set) >= 2:
            vmap[v] = new_set
        else:
            vmap.pop(v, None)
            if not vmap:
                del self.conflict_cells[t]


@dataclass(frozen=True)
class CBSNode:
    """A high-level CBS search-tree node.

    ``constraints[k] = (neg, pos)``: agent ``k`` must avoid every cell in
    ``neg`` and must occupy every cell in ``pos`` (one positive per time-step;
    contradictions are caught by the constrained smart sweep returning
    ``None``).

    ``conflicts`` carries an incremental conflict-cell map shared across
    every per-pop query (bypass check, cardinal selection, non-cardinal
    branching). Child nodes inherit the parent's map and update in
    ``O(horizon)`` per replaced agent rather than re-scanning every
    timestep from scratch.
    """

    lb: int
    constraints: tuple[NegPos, ...]
    plans: tuple[tuple[Any, ...], ...]
    per_agent_cost: tuple[int, ...]
    node_id: int
    conflicts: _ConflictMap
    parent_id: int | None = None


@dataclass(frozen=True)
class _LazyChild:
    """A child whose constrained-sweep solves are deferred until it is popped.

    Pushed onto the open list under ``push_lb`` (the parent's LB, a sound
    lower bound on the child's subtree optimum since adding constraints
    cannot lower any agent's per-agent cost). On pop, the deferred
    ``cached_solve`` calls are run, the true LB is computed, and the
    resolved :class:`CBSNode` either (i) is dropped as infeasible, (ii) is
    re-pushed under its true LB if that LB exceeds ``push_lb``, or
    (iii) proceeds straight to bypass / expansion. Saves the constrained-DP
    cost on every child whose subtree is pruned before it would have been
    popped — common on hard instances where bypass closes the search shortly
    after the optimal LB is hit.
    """

    push_lb: int
    parent: CBSNode
    constraint_updates: tuple[tuple[int, frozenset[Cell], frozenset[Cell]], ...]
    node_id: int


def _iter_vertex_conflicts(
    plans: Sequence[Sequence[V]],
) -> itertools.chain[Conflict]:
    """Every pairwise vertex conflict in earliest-time-then-lex order.

    A vertex conflict is ``(i, j, v, t)`` with ``i < j`` and
    ``plans[i][t] == plans[j][t] == v``. Trajectories of unequal length are
    padded with the agent's last vertex; empty plans are skipped.

    Kept as a standalone helper for external callers (property tests,
    diagnostic scripts). Internal :func:`solve_ccbs` uses
    :class:`_ConflictMap` instead.
    """
    if not plans:
        return iter(())  # type: ignore[return-value]
    n = len(plans)
    horizon = max((len(p) for p in plans), default=0)

    def gen() -> itertools.chain[Conflict]:
        for t in range(horizon):
            cell_to_agents: dict[Hashable, list[int]] = {}
            for i in range(n):
                p = plans[i]
                if not p:
                    continue
                v: Hashable = p[t] if t < len(p) else p[-1]
                cell_to_agents.setdefault(v, []).append(i)
            for v, agents in cell_to_agents.items():
                if len(agents) < 2:
                    continue
                for x in range(len(agents)):
                    for y in range(x + 1, len(agents)):
                        yield Conflict(i=agents[x], j=agents[y], v=v, t=t)

    return gen()  # type: ignore[return-value]


def find_vertex_conflict(
    plans: Sequence[Sequence[V]],
) -> Conflict | None:
    """Return the earliest vertex conflict in a joint plan, or ``None``.

    Kept for callers that want a single conflict without classifying every
    candidate; ``solve_ccbs`` internally uses cardinal-first selection
    over a :class:`_ConflictMap`.
    """
    for c in _iter_vertex_conflicts(plans):
        return c
    return None


CachedSolve = Callable[
    [int, frozenset[Cell], frozenset[Cell]],
    tuple[int, tuple[Any, ...]] | None,
]


def _classify_conflict(
    node: CBSNode,
    conflict: Conflict,
    cached_solve: CachedSolve,
) -> tuple[str, float]:
    """Classify ``conflict`` as cardinal / semi / non per App. A.2.

    Returns ``(class, score)`` where ``score = max(d_i, d_j)`` ranks
    candidates within a class. ``d_i, d_j`` are the per-agent LB increases
    from adding ``(conflict.v, conflict.t)`` to each agent's *negative*
    constraints; infeasibility maps to ``+inf``. Positive constraints are
    left untouched in the lookahead — the existing positive constraints of
    each agent stay in place.
    """
    new_cell: Cell = (conflict.v, conflict.t)
    neg_i, pos_i = node.constraints[conflict.i]
    neg_j, pos_j = node.constraints[conflict.j]
    result_i = cached_solve(conflict.i, neg_i | {new_cell}, pos_i)
    result_j = cached_solve(conflict.j, neg_j | {new_cell}, pos_j)

    d_i = _INF if result_i is None else result_i[0] - node.per_agent_cost[conflict.i]
    d_j = _INF if result_j is None else result_j[0] - node.per_agent_cost[conflict.j]

    if d_i > 0 and d_j > 0:
        cls = "cardinal"
    elif d_i > 0 or d_j > 0:
        cls = "semi"
    else:
        cls = "non"
    return cls, max(d_i, d_j)


def _select_conflict(
    node: CBSNode,
    cached_solve: CachedSolve,
) -> Conflict | None:
    """App. A.2: prefer cardinal > semi > non; tie-break by score then order.

    Iterates conflicts off the node's incremental conflict map in
    earliest-time-then-lex order. Returns the first cardinal conflict
    found (short-circuit), otherwise the first semi, else the first non.
    """
    first_semi: Conflict | None = None
    first_non: Conflict | None = None
    for conflict in node.conflicts.iter_conflicts_sorted():
        cls, _ = _classify_conflict(node, conflict, cached_solve)
        if cls == "cardinal":
            return conflict
        if cls == "semi":
            if first_semi is None:
                first_semi = conflict
        else:
            if first_non is None:
                first_non = conflict
    return first_semi if first_semi is not None else first_non


def _count_conflicts(plans: Sequence[Sequence[V]]) -> int:
    """Number of pairwise vertex conflicts in a joint plan.

    Used by App. A.5 bypass to compare a candidate child to its parent.
    """
    return sum(1 for _ in _iter_vertex_conflicts(plans))


def _pick_split_agent(node: CBSNode, conflict: Conflict) -> int:
    """Pick which conflict-agent gets the (positive, negative) disjoint split.

    Heuristic: split on the agent with fewer existing constraints (cheaper
    to constrain further); ties go to the lower index. Matches the
    CBS-PC literature's "fewer plans, more constraining" rationale.
    """
    neg_i, pos_i = node.constraints[conflict.i]
    neg_j, pos_j = node.constraints[conflict.j]
    size_i = len(neg_i) + len(pos_i)
    size_j = len(neg_j) + len(pos_j)
    if size_i <= size_j:
        return conflict.i
    return conflict.j


def solve_ccbs(
    schedules: Sequence[Sequence[V]],
    *,
    time_limit_sec: float | None = None,
    cardinal_trigger_pops: int = 0,
    use_disjoint_split: bool = True,
    use_bypass: bool = True,
    diagnostics: dict[str, Any] | None = None,
    **_: object,
) -> JointPlan | None:
    """Exact CBS-style joint solver for a MAPF-Collapse sub-instance.

    Operates on a single non-trivial component (the caller is responsible for
    component decomposition; singletons should be dispatched directly to
    :func:`mapfc.single_agent.collapse_dag.solve_unconstrained` instead).

    Returns ``None`` if the time limit is exceeded before any conflict-free
    node is popped. Returns a degenerate :class:`JointPlan` of empty plans
    when called on an empty input.

    ``cardinal_trigger_pops`` controls the adaptive A.2 selector: as long as
    the cumulative high-level pop count is at or below this threshold, the
    cheap earliest-time conflict is branched on; once it exceeds, cardinal
    classification (Alg.~\\ref{alg:eng-cardinal}) kicks in for the remainder
    of the search. Pass ``0`` to force cardinal-first from the root, or a
    very large value to disable A.2 entirely.

    ``use_disjoint_split`` toggles App. A.4: when True (default), each
    conflict branches ``(positive, negative)`` on a chosen agent and the
    "other" agent gets an implied negative constraint; when False, the
    classical CBS branch ``(negative_i, negative_j)`` is used. Disjoint
    splitting produces disjoint search subspaces and propagates positive
    constraints to the other conflict agent — typically a large speedup.

    ``use_bypass`` toggles App. A.5: as each child is generated, if its
    joint plan is already conflict-free AND its cost equals the parent's
    lower bound, the child IS the joint optimum (parent's LB is a valid
    lower bound on the subtree's optimum, so a feasible plan at LB is
    optimal) and is returned immediately, short-circuiting the rest of
    the search. The ICBS variant that *discards* the sibling under a
    semi-cardinal condition is **not** sound in MAPF-Collapse because
    the LB (sum of per-agent constrained optima) is typically not tight
    — an optimum can sit strictly above the parent's LB, in which case
    discarding the higher-LB sibling can lose it. We exclude that
    aggressive variant; see DECISIONS.md for the A/B evidence.

    If a mutable ``diagnostics`` dict is supplied, it is populated on return
    with ``pop_count``, ``cardinal_activated``, and ``bypass_count`` so
    callers can verify the adaptive switch and bypass behaviour in tests.
    """
    t0 = time.monotonic()
    if not schedules:
        return JointPlan(plans=(), cost=0, solver="ccbs", runtime_s=0.0, build_runtime_s=0.0)

    horizon = max(len(s) for s in schedules)
    padded: list[tuple[V, ...]] = []
    original_lengths: list[int] = []
    for sched in schedules:
        s_list = list(sched)
        original_lengths.append(len(s_list))
        if not s_list:
            padded.append(tuple(s_list))
            continue
        last = s_list[-1]
        while len(s_list) < horizon:
            s_list.append(last)
        padded.append(tuple(s_list))

    cache: dict[
        tuple[int, frozenset[Cell], frozenset[Cell]],
        tuple[int, tuple[Any, ...]] | None,
    ] = {}

    def cached_solve(
        agent_idx: int,
        negative: frozenset[Cell],
        positive: frozenset[Cell],
    ) -> tuple[int, tuple[Any, ...]] | None:
        key = (agent_idx, negative, positive)
        if key not in cache:
            r = solve_constrained(padded[agent_idx], negative, positive)
            cache[key] = (r[0], tuple(r[1])) if r is not None else None
        return cache[key]

    id_gen = itertools.count()
    empty_C: frozenset[Cell] = frozenset()
    empty_pair: NegPos = (empty_C, empty_C)
    root_costs: list[int] = []
    root_plans: list[tuple[V, ...]] = []
    for i, sched in enumerate(padded):
        c, p = solve_unconstrained(sched)
        cache[(i, empty_C, empty_C)] = (c, tuple(p))
        root_costs.append(c)
        root_plans.append(tuple(p))

    root_plans_tuple = tuple(root_plans)
    root = CBSNode(
        lb=sum(root_costs),
        constraints=(empty_pair,) * len(padded),
        plans=root_plans_tuple,
        per_agent_cost=tuple(root_costs),
        node_id=next(id_gen),
        conflicts=_ConflictMap.from_plans(root_plans_tuple),
        parent_id=None,
    )
    open_list: list[tuple[int, int, CBSNode | _LazyChild]] = []
    heapq.heappush(open_list, (root.lb, root.node_id, root))

    pop_count = 0
    cardinal_active = False
    bypass_count = 0

    def _resolve_lazy(item: _LazyChild) -> CBSNode | None:
        """Run the deferred ``cached_solve`` calls for ``item``'s constraint
        updates and assemble the full :class:`CBSNode`. Returns ``None`` if
        any constrained solve is infeasible — caller drops the child.
        """
        parent = item.parent
        new_constraints = list(parent.constraints)
        new_costs = list(parent.per_agent_cost)
        new_plans = list(parent.plans)
        changed: list[tuple[int, tuple[Any, ...], tuple[Any, ...]]] = []
        for agent_idx, neg, pos in item.constraint_updates:
            result = cached_solve(agent_idx, neg, pos)
            if result is None:
                return None
            c_new, pi_new = result
            new_constraints[agent_idx] = (neg, pos)
            new_costs[agent_idx] = c_new
            changed.append((agent_idx, new_plans[agent_idx], pi_new))
            new_plans[agent_idx] = pi_new
        child_conflicts = parent.conflicts.clone()
        for agent_idx, old_p, new_p in changed:
            child_conflicts.replace_agent(agent_idx, old_p, new_p)
        return CBSNode(
            lb=sum(new_costs),
            constraints=tuple(new_constraints),
            plans=tuple(new_plans),
            per_agent_cost=tuple(new_costs),
            node_id=item.node_id,
            conflicts=child_conflicts,
            parent_id=parent.node_id,
        )

    def _check_bypass_optimum(parent: CBSNode, child: CBSNode) -> JointPlan | None:
        """App. A.5: child is the joint optimum if its plan is conflict-free
        AND its cost equals the parent's LB (a sound lower bound — a feasible
        plan at LB cost is optimal). Returns the wrapped :class:`JointPlan`
        on hit, ``None`` otherwise."""
        if not use_bypass:
            return None
        if child.lb != parent.lb:
            return None
        if child.conflicts.has_conflict():
            return None
        nonlocal bypass_count
        bypass_count += 1
        trimmed = tuple(plan[: original_lengths[i]] for i, plan in enumerate(child.plans))
        if diagnostics is not None:
            diagnostics["pop_count"] = pop_count
            diagnostics["cardinal_activated"] = cardinal_active
            diagnostics["bypass_count"] = bypass_count
        return JointPlan(
            plans=trimmed,
            cost=sum(child.per_agent_cost),
            solver="ccbs",
            runtime_s=time.monotonic() - t0,
            build_runtime_s=0.0,
        )

    while open_list:
        if time_limit_sec is not None and time.monotonic() - t0 > time_limit_sec:
            if diagnostics is not None:
                diagnostics["pop_count"] = pop_count
                diagnostics["cardinal_activated"] = cardinal_active
                diagnostics["bypass_count"] = bypass_count
            return None
        _, _, item = heapq.heappop(open_list)
        if isinstance(item, _LazyChild):
            resolved = _resolve_lazy(item)
            if resolved is None:
                continue
            bypass_result = _check_bypass_optimum(item.parent, resolved)
            if bypass_result is not None:
                return bypass_result
            if resolved.lb > item.push_lb:
                heapq.heappush(open_list, (resolved.lb, resolved.node_id, resolved))
                continue
            node = resolved
        else:
            node = item
        pop_count += 1
        if not cardinal_active and pop_count > cardinal_trigger_pops:
            cardinal_active = True
        if cardinal_active:
            conflict = _select_conflict(node, cached_solve)
        else:
            conflict = node.conflicts.first_conflict()
        if conflict is None:
            trimmed_plans = tuple(plan[: original_lengths[i]] for i, plan in enumerate(node.plans))
            if diagnostics is not None:
                diagnostics["pop_count"] = pop_count
                diagnostics["cardinal_activated"] = cardinal_active
                diagnostics["bypass_count"] = bypass_count
            return JointPlan(
                plans=trimmed_plans,
                cost=sum(node.per_agent_cost),
                solver="ccbs",
                runtime_s=time.monotonic() - t0,
                build_runtime_s=0.0,
            )

        new_cell: Cell = (conflict.v, conflict.t)
        # Lazy LB: push every child as a :class:`_LazyChild` keyed by the
        # parent's LB (a sound lower bound on the child's true LB, since
        # adding constraints never lowers any agent's per-agent cost). The
        # deferred ``cached_solve`` calls only run when the child is popped,
        # so children whose subtree is pruned earlier in the search never
        # pay the constrained-DP cost. Bypass classification (App. A.5)
        # likewise moves to pop time, after resolution.
        lazy_children: list[_LazyChild] = []
        if use_disjoint_split:
            split_agent = _pick_split_agent(node, conflict)
            split_neg, split_pos = node.constraints[split_agent]
            # All OTHER agents whose current plan also has the cell at the
            # conflict time — they all need (v, t) added to their negatives
            # in the positive branch.
            t_c = conflict.t
            v_c = conflict.v
            others_at_cell: list[int] = [
                k
                for k in range(len(node.plans))
                if k != split_agent and len(node.plans[k]) > t_c and node.plans[k][t_c] == v_c
            ]

            # Left child: split_agent gets a negative constraint; nothing else
            # changes.
            left_neg = split_neg | {new_cell}
            lazy_children.append(
                _LazyChild(
                    push_lb=node.lb,
                    parent=node,
                    constraint_updates=((split_agent, left_neg, split_pos),),
                    node_id=next(id_gen),
                )
            )

            # Right child: split_agent gets a positive; every other agent
            # currently at (v, t) gets the matching negative.
            right_pos = split_pos | {new_cell}
            right_updates: list[tuple[int, frozenset[Cell], frozenset[Cell]]] = [
                (split_agent, split_neg, right_pos)
            ]
            for k in others_at_cell:
                neg_k, pos_k = node.constraints[k]
                right_updates.append((k, neg_k | {new_cell}, pos_k))
            lazy_children.append(
                _LazyChild(
                    push_lb=node.lb,
                    parent=node,
                    constraint_updates=tuple(right_updates),
                    node_id=next(id_gen),
                )
            )
        else:
            for k in (conflict.i, conflict.j):
                neg_k, pos_k = node.constraints[k]
                lazy_children.append(
                    _LazyChild(
                        push_lb=node.lb,
                        parent=node,
                        constraint_updates=((k, neg_k | {new_cell}, pos_k),),
                        node_id=next(id_gen),
                    )
                )

        for child in lazy_children:
            heapq.heappush(open_list, (child.push_lb, child.node_id, child))

    if diagnostics is not None:
        diagnostics["pop_count"] = pop_count
        diagnostics["cardinal_activated"] = cardinal_active
        diagnostics["bypass_count"] = bypass_count
    return None
