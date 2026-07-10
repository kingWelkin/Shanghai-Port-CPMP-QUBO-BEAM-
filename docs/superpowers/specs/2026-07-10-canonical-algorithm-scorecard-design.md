# Canonical Algorithm Scorecard Design

Date: 2026-07-10
Status: Approved approach, pending implementation

## Goal

Create one repository-tracked, prominently linked comparison cluster for the
four benchmark surfaces used to judge the project:

- Real30
- Open120
- Paper BF721
- Paper CV840

The scorecard compares the promoted QUBO+Beam mainline with paper Wang BS-B,
paper BS-FRG, A*, and classic beam. Every cell reports solved coverage and the
sum of relocation moves over solved cases. The scorecard is a comparison
artifact, not a replacement for the raw per-case CSVs.

## Canonical Values

| Algorithm | Real30 | Open120 | BF | CV |
|---|---:|---:|---:|---:|
| QUBO+Beam selector v4.3 | 30/30, 525 | 120/120, 3,759 | 721/721, 45,906 | 840/840, 55,865 |
| Paper Wang BS-B | 30/30, 757 | 120/120, 4,569 | 141/721, 5,533 | 762/840, 27,117 |
| Paper BS-FRG | 30/30, 516 | 120/120, 4,121 | 719/721, 51,509 | 826/840, 46,299 |
| A* no-proof | 30/30, 514 | N/A | N/A | N/A |
| Classic beam | 28/30, 455 | 84/120, 2,711 | 642/721, 37,180 | 668/840, 25,312 |

The v4.3 Real30 value is fixed at `525`, matching the promoted raw result CSV.
The `516` value belongs to paper BS-FRG and must not be reused for v4.3.

## Metric Contract

Each algorithm/suite observation contains:

- `solved`: number of accepted solved cases;
- `total`: number of cases in the suite;
- `moves_solved_sum`: relocation moves summed only over solved cases;
- `applicability`: `measured` or `not_applicable`;
- `time_limit_sec`: the benchmark budget used by the source run;
- `source_artifact`: repository-relative raw CSV or summary path;
- `source_scope`: note describing any historical-control or no-proof boundary;
- `evidence_status`: `raw`, `derived`, or `frozen_reference`.

`moves_solved_sum` is not a penalty for unsolved cases. Coverage must therefore
be read before comparing move totals. `N/A` means the algorithm was not treated
as scalable/applicable on that benchmark, not that it solved zero cases.

## Artifact Layout

Create:

```text
docs/experiments/canonical_algorithm_scorecard_v43/
|-- README.md
|-- canonical_algorithm_scorecard.csv
|-- source_manifest.csv
|-- canonical_algorithm_scorecard_zh.png
`-- sources/
    |-- bs_frg_pure_paper_full_20260618.csv
    `-- bs_frg_pure_paper_full_20260618_summary.md
```

The local-only pure BS-FRG paper full run is copied into `sources/` because it
is necessary to audit the split BF/CV values. Other evidence remains at its
existing tracked path and is referenced rather than duplicated.

## Evidence Mapping

- v4.3 Open120: `docs/experiments/v43_wang_reintegration_20260709/`
  `v43_wang_route_open120_t120_all_dlts_j8.csv`.
- v4.3 BF/CV: `docs/experiments/v43_full_gate_20260709/`
  `v43_wang_route_paper_bf_cv_t120_j8.csv`.
- v4.3 Real30: `docs/experiments/v43_full_gate_20260709/`
  `v43_real30_mainline_branch_t120.csv`.
- Paper Wang Open120 and BF/CV: the tracked `paper_wang_bs_b_*` files under
  `docs/experiments/v43_wang_reintegration_20260709/`.
- Paper Wang Real30, paper BS-FRG Open120/Real30, classic Open120/BF/CV, and the
  current comparison derivation: tracked files under
  `docs/reports/hardness_quality_all_cases_20260709/`.
- Paper BS-FRG BF/CV: copied raw full-run CSV and summary in the canonical
  package `sources/` directory.
- A* and classic Real30:
  `docs/experiments/astar_qubo_baseline_beam_comparison.csv`, using the
  `astar_noproof_*` and `baseline_beam_budget_best_*` columns respectively.
  The manifest records the no-proof/historical-control boundary.

## Generated Visual

Add a deterministic builder script that reads
`canonical_algorithm_scorecard.csv` and generates the Chinese PNG. The PNG
keeps the supplied layout semantics:

- algorithms as rows and Real30/Open120/BF/CV as columns;
- solved coverage on the first line;
- solved-case move sum on the second line;
- green for `100%` coverage, yellow for `[98%, 100%)`, orange for
  `[90%, 98%)`, red for below `90%`, and gray for N/A;
- the promoted QUBO+Beam row visually emphasized without changing its values.

The PNG must not contain independently hard-coded numbers. CSV is the only
input, so table and image cannot drift.

## Highlight Locations

Add a compact scorecard or direct link near the top of:

- root `README.md`;
- `docs/index.md`;
- `docs/experiments/README.md`;
- `docs/reports/README.md`.

The existing v4.3 mainline statement remains authoritative. The comparison
cluster is the authoritative cross-algorithm evidence surface.

## Validation

- Recompute every measured row from its source CSV where raw data is present.
- Assert the exact solved/moves values in the canonical CSV.
- Assert all source-manifest paths exist and are Git tracked.
- Assert the pure BS-FRG paper source copy matches the local source SHA-256.
- Regenerate the PNG and visually inspect it for clipping and incorrect color
  states.
- Run Markdown local-link checks and the full test suite.
- Confirm the canonical directory and every highlighted entry are tracked
  before commit and push.

## Non-Goals

- Do not rerun the benchmark suites.
- Do not change the promoted v4.3 algorithm or solver parameters.
- Do not promote paper baselines to mainline status.
- Do not import the hundreds of unrelated historical scratch experiments.
