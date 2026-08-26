"""Read Tang et al.'s POGEMA-derived MAPF datasets and yield instances.

Each dataset directory ships:
- ``maps.yaml`` — a dict mapping ``map_name -> ASCII grid string``.
- ``<dataset>-mapf.yaml`` — a config with ``environment.num_agents.grid_search``
  (the list of agent counts to expand each scenario into) and ``scenarios``
  (a dict keyed by ``Scenario-NNN``, each carrying ``agents_xy``, ``targets_xy``,
  ``seed``, ``map_name`` at the maximum-agents grid value).

For a fixed scenario S with maximum-agent count K, the loader emits one
``Instance`` per ``k`` in the grid, using the first ``k`` entries of
``S.agents_xy`` and ``S.targets_xy``. The product across all scenarios and
grid values matches Tang et al.'s per-dataset count:

==============  =========  ====================  =========
dataset         scenarios  num_agents grid       instances
==============  =========  ====================  =========
01-random       128        [8,16,24,32,48,64]    768
02-mazes        128        [8,16,24,32,48,64]    768
03-warehouse    128        [32,64,96,128,160,    768
                            192]
04-movingai     128        [64,128,192,256]      512
05-puzzles      160        [2,3,4]               480
==============  =========  ====================  =========

The 5-dataset total is 3,296 — matches ``third_party/SUBMODULE_SHAS.json``.

Out of scope:
``06-pathfinding`` is single-agent pathfinding (80 scenarios, num_agents=1)
and is NOT a MAPF instance set; the loader refuses to load it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

DATASETS: Final[frozenset[str]] = frozenset(
    {"01-random", "02-mazes", "03-warehouse", "04-movingai", "05-puzzles"}
)


@dataclass(frozen=True)
class Instance:
    """One MAPF instance: a (scenario, num_agents) pair instantiated against a map."""

    dataset: str
    scenario_name: str
    map_name: str
    num_agents: int
    seed: int
    grid: tuple[str, ...]
    agents_xy: tuple[tuple[int, int], ...]
    targets_xy: tuple[tuple[int, int], ...]

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    @property
    def instance_id(self) -> str:
        return f"{self.dataset}/{self.scenario_name}/n{self.num_agents}"


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not parse to a mapping (got {type(data).__name__})")
    return data


def _normalize_grid(grid: str) -> tuple[str, ...]:
    lines = grid.split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    return tuple(lines)


def _grid_search(num_agents_field: dict[str, Any] | int) -> list[int]:
    if isinstance(num_agents_field, int):
        return [num_agents_field]
    if isinstance(num_agents_field, dict) and "grid_search" in num_agents_field:
        return [int(k) for k in num_agents_field["grid_search"]]
    raise ValueError(
        f"unexpected environment.num_agents shape: {num_agents_field!r}; expected an int "
        "or a dict with a 'grid_search' list"
    )


def load_dataset(dataset_path: Path) -> Iterator[Instance]:
    """Yield every MAPF instance in ``dataset_path`` deterministically.

    Scenarios are emitted in sorted-key order; within a scenario, ``num_agents``
    values follow ``environment.num_agents.grid_search`` order.

    Parameters
    ----------
    dataset_path
        Path to one of the five supported dataset directories, e.g.
        ``third_party/judgelight/pogema-benchmark/algorithms/experiments/02-mazes``.

    Raises
    ------
    ValueError
        If the directory name is not one of the five MAPF datasets, or if
        required YAML keys are missing.
    """
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.name
    if dataset_name not in DATASETS:
        raise ValueError(
            f"unsupported dataset {dataset_name!r}; expected one of {sorted(DATASETS)}"
        )

    maps_yaml = dataset_path / "maps.yaml"
    mapf_yaml = dataset_path / f"{dataset_name}-mapf.yaml"
    if not maps_yaml.exists():
        raise FileNotFoundError(f"{maps_yaml} not found")
    if not mapf_yaml.exists():
        raise FileNotFoundError(f"{mapf_yaml} not found")

    maps: dict[str, str] = _load_yaml(maps_yaml)
    config: dict[str, Any] = _load_yaml(mapf_yaml)
    environment: dict[str, Any] = config["environment"]
    scenarios: dict[str, Any] = config["scenarios"]
    agent_grid: list[int] = _grid_search(environment["num_agents"])

    for scenario_name in sorted(scenarios):
        scn = scenarios[scenario_name]
        map_name = scn["map_name"]
        seed = int(scn["seed"])
        agents_xy_all = scn["agents_xy"]
        targets_xy_all = scn["targets_xy"]
        max_agents = len(agents_xy_all)
        if len(targets_xy_all) != max_agents:
            raise ValueError(
                f"{dataset_name}/{scenario_name}: agents_xy and targets_xy length "
                f"mismatch ({len(agents_xy_all)} vs {len(targets_xy_all)})"
            )

        grid = _normalize_grid(maps[map_name])

        for k in agent_grid:
            if k > max_agents:
                continue
            yield Instance(
                dataset=dataset_name,
                scenario_name=scenario_name,
                map_name=map_name,
                num_agents=k,
                seed=seed,
                grid=grid,
                agents_xy=tuple((int(x), int(y)) for x, y in agents_xy_all[:k]),
                targets_xy=tuple((int(x), int(y)) for x, y in targets_xy_all[:k]),
            )


def count_instances(dataset_path: Path) -> int:
    """Materialise the iterator and return the instance count.

    Convenience for spot-checks against the totals in
    ``third_party/SUBMODULE_SHAS.json``.
    """
    return sum(1 for _ in load_dataset(dataset_path))
