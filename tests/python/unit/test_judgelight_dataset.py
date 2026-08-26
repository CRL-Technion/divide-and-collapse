"""Unit tests for mapfc.io.judgelight_dataset.

Skip every test if the vendored Judgelight checkout is missing; CI installs
the fast-path lint+test job without cloning Judgelight, so these tests are
purely local.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mapfc.io.judgelight_dataset import DATASETS, Instance, count_instances, load_dataset

JUDGELIGHT_EXPERIMENTS = (
    Path(__file__).resolve().parents[3]
    / "third_party"
    / "judgelight"
    / "pogema-benchmark"
    / "algorithms"
    / "experiments"
)

pytestmark = pytest.mark.skipif(
    not JUDGELIGHT_EXPERIMENTS.exists(),
    reason="third_party/judgelight not present; run scripts/setup_judgelight.{sh,ps1}",
)


EXPECTED_INSTANCE_COUNTS = {
    "01-random": 768,
    "02-mazes": 768,
    "03-warehouse": 768,
    "04-movingai": 512,
    "05-puzzles": 480,
}


@pytest.mark.parametrize("dataset", sorted(DATASETS))
def test_instance_counts_match_briefing(dataset: str) -> None:
    path = JUDGELIGHT_EXPERIMENTS / dataset
    assert count_instances(path) == EXPECTED_INSTANCE_COUNTS[dataset]


def test_rejects_unknown_dataset(tmp_path: Path) -> None:
    (tmp_path / "07-unknown").mkdir()
    with pytest.raises(ValueError, match="unsupported dataset"):
        next(load_dataset(tmp_path / "07-unknown"))


def test_rejects_pathfinding_dataset() -> None:
    pathfinding = JUDGELIGHT_EXPERIMENTS / "06-pathfinding"
    if not pathfinding.exists():
        pytest.skip("06-pathfinding not present")
    with pytest.raises(ValueError, match="unsupported dataset"):
        next(load_dataset(pathfinding))


def test_01_random_first_instance_shape() -> None:
    first = next(load_dataset(JUDGELIGHT_EXPERIMENTS / "01-random"))
    assert isinstance(first, Instance)
    assert first.dataset == "01-random"
    assert first.scenario_name.startswith("Scenario-")
    assert first.num_agents == 8
    assert len(first.agents_xy) == 8
    assert len(first.targets_xy) == 8
    assert first.height > 0 and first.width > 0


def test_05_puzzles_grid_search_order() -> None:
    instances = list(load_dataset(JUDGELIGHT_EXPERIMENTS / "05-puzzles"))
    by_scenario: dict[str, list[int]] = {}
    for inst in instances:
        by_scenario.setdefault(inst.scenario_name, []).append(inst.num_agents)
    sample = by_scenario[next(iter(by_scenario))]
    assert sample == [2, 3, 4]


def test_load_is_deterministic() -> None:
    path = JUDGELIGHT_EXPERIMENTS / "05-puzzles"
    a = [(i.scenario_name, i.num_agents) for i in load_dataset(path)]
    b = [(i.scenario_name, i.num_agents) for i in load_dataset(path)]
    assert a == b


def test_agents_truncated_to_num_agents() -> None:
    by_count: dict[int, Instance] = {}
    for inst in load_dataset(JUDGELIGHT_EXPERIMENTS / "01-random"):
        if inst.scenario_name == "Scenario-640":
            by_count[inst.num_agents] = inst
    assert set(by_count) == {8, 16, 24, 32, 48, 64}
    for k, inst in by_count.items():
        assert len(inst.agents_xy) == k
        assert len(inst.targets_xy) == k
    bigger = by_count[64].agents_xy
    for k, inst in by_count.items():
        assert inst.agents_xy == bigger[:k]
        assert inst.targets_xy == by_count[64].targets_xy[:k]
