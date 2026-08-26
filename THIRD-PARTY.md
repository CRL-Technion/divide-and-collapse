# Third-party components

This repository contains **no third-party code and no third-party data**.
Everything under version control here is original work released under the MIT
License (see `LICENSE`). Two external projects are used by the evaluation, and
both are fetched or regenerated on your machine rather than redistributed.

## Judgelight (Tang, Koenig, and Biyik)

The baseline solver, and the reference implementation of the MAPF-Collapse
problem this work builds on.

- Upstream: <https://github.com/TachikakaMin/Judgelight>
- Paper: Tang, Koenig, and Biyik. *Judgelight: Trajectory-Level
  Post-Optimization for Multi-Agent Path Finding via Closed-Subwalk
  Collapsing.* WAFR 2026.
- How we use it: fetched on demand into `third_party/judgelight/` by
  `scripts/setup_judgelight.sh` (or `.ps1`), pinned to the commit SHAs recorded
  in `third_party/SUBMODULE_SHAS.json`. That directory is git-ignored here.

**Licensing note.** At the time of writing, the Judgelight repository publishes
no license file, so no redistribution rights are granted by default. We
therefore never vendor, mirror, or re-publish any part of it: you fetch it
yourself, directly from its authors, under whatever terms they offer. Every
configuration whose name contains `judgelight` requires that fetch (and a
Gurobi installation); all other configurations run without it.

## POGEMA (Skrynnik et al.)

The benchmark platform whose scenarios generate our input plans.

- Upstream: <https://github.com/AIRI-Institute/pogema>
- Paper: Skrynnik, Andreychuk, Borzilov, Chernyavskiy, Yakovlev, and Panov.
  *POGEMA: A Benchmark Platform for Cooperative Multi-Agent Pathfinding.*
  ICLR 2025.
- License: MIT (Copyright (c) 2022, Alexey Skrynnik).
- How we use it: the input schedule of each instance is produced by running
  POGEMA's `BatchAStarAgent` over the POGEMA scenario suite. We ship none of
  the resulting data. Regenerate it with `scripts/dump_pogema_full.py`, which
  is deterministic (pinned SHAs, seed 0), so anyone can rebuild the exact
  3,296-instance input set we evaluated on.

Although POGEMA's MIT license would permit redistribution with attribution, we
prefer to keep this repository free of other people's data and have you fetch
it from the source.

## Examples

The instances in `examples/data/` are ours: they are written from scratch by
`examples/data/make_examples.py` and derive from no external benchmark.
