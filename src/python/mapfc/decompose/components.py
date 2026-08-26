"""Union-find connected components on the interaction graph (Sec. 4).

Standard union-by-rank + path-compression union-find. Returns one
``frozenset[int]`` per connected component of H, plus an explicit
``component_of: tuple[int, ...]`` array that maps each agent to its
component index for ``O(1)`` lookups by downstream callers.
"""

from __future__ import annotations

from collections.abc import Iterable


def connected_components(
    n: int,
    edges: Iterable[frozenset[int]],
) -> tuple[list[frozenset[int]], tuple[int, ...]]:
    """Return ``(components, component_of)`` of the graph ``(n, edges)``.

    ``components`` is a list of disjoint ``frozenset[int]`` covering
    ``{0, ..., n-1}``, sorted by ``(size, sorted-tuple-of-members)`` so the
    output is deterministic across runs. ``component_of[i]`` is the index of
    the component containing agent ``i``.
    """
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    for edge in edges:
        it = iter(edge)
        a = next(it)
        b = next(it)
        union(a, b)

    by_root: dict[int, list[int]] = {}
    for x in range(n):
        by_root.setdefault(find(x), []).append(x)

    components = sorted(
        (frozenset(members) for members in by_root.values()),
        key=lambda s: (len(s), tuple(sorted(s))),
    )
    component_of_arr = [-1] * n
    for idx, comp in enumerate(components):
        for agent in comp:
            component_of_arr[agent] = idx
    return components, tuple(component_of_arr)
