# v4.0 Scorecard And Reproduction

Date: 2026-07-03
Commit: `411da4b Add stable chainaware quality experiment`

## Promoted Profile

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish
```

## Fresh Full-Surface Scorecard

No business cap, `time_limit=120`, `alpha=10`, `jobs=8`.

| benchmark surface | cases | solved | moves | full penalty | avg runtime sec | max moves |
|---|---:|---:|---:|---:|---:|---:|
| open120 DLTS | 120 | 120 | 6928 | 6928 | 0.882 | 218 |
| paper BF/CV full1561 | 1561 | 1561 | 106903 | 106903 | 3.933 | 670 |
| paper BF | 721 | 721 | 46517 | 46517 | 5.786 | - |
| paper CV | 840 | 840 | 60386 | 60386 | 2.343 | - |

## Open120 Breakdown

| family | cases | valid | moves | avg runtime sec |
|---|---:|---:|---:|---:|
| 5x5_cv1_test | 20 | 20 | 1224 | 0.642 |
| 5x5_cv2_test | 20 | 20 | 1297 | 0.698 |
| 5x5_cv3_test | 20 | 20 | 1168 | 0.245 |
| 7x5_cv1_test | 20 | 20 | 1129 | 1.638 |
| 7x5_cv2_test | 20 | 20 | 1096 | 0.967 |
| 7x5_cv3_test | 20 | 20 | 1014 | 1.102 |

## Hard Paper Families

The full BF/CV run solved all rows, including the hard families that previously
caused trace work:

| family | solved | moves | avg runtime sec |
|---|---:|---:|---:|
| BF14 | 20/20 | 2622 | 10.781 |
| BF16 | 20/20 | 2667 | 10.067 |
| BF30 | 20/20 | 2714 | 13.810 |
| BF32 | 20/20 | 2699 | 14.618 |
| CV/10-6 | 40/40 | 15152 | 5.822 |
| CV/10-10 | 40/40 | 13475 | 20.944 |

## Reproduction Commands

From the repository root or this worktree:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish --time-limit 120 --alpha 10 --jobs 8 --no-business-cap --output docs\experiments\v4_open120_t120.csv
```

```powershell
python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF CV --sample-per-family 9999 --variant qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish --time-limit 120 --alpha 10 --jobs 8 --no-business-cap --output docs\experiments\v4_paper_bf_cv_full1561_t120.csv
```

If running from a worktree that does not contain `external`, use absolute source
paths to the main checkout's data directories.

## Source Artifacts

The current scorecard is based on:

- `docs/experiments/chainaware_commit411da4b_open120_t120_20260703.csv`
- `docs/experiments/chainaware_commit411da4b_open120_t120_20260703_summary.md`
- `docs/experiments/chainaware_commit411da4b_paper_bf_cv_full1561_t120_20260703.csv`
- `docs/experiments/chainaware_commit411da4b_paper_bf_cv_full1561_t120_20260703_summary.md`

There is also a 611-row paper side run from the runner's default sample setting.
It is useful as a smoke result, but it is not the full BF/CV scorecard.

## Promotion Checklist

Use this checklist before replacing v4.0:

- open120 no-cap t120 remains `120/120`;
- paper BF full set remains fully solved;
- paper CV full set remains fully solved;
- full paper command includes `--sample-per-family 9999`;
- moves, full penalty, and runtime are reported for open120, BF, and CV
  separately;
- no result is promoted from a patched aggregate when a direct full rerun is
  feasible;
- the mechanism gain is attributed using telemetry, not just final CSV totals.

## Current Quality Caveat

v4.0 is the current robustness mainline. It may not be the shortest solver on
every common-solved CV row compared with pure BS-FRG. The next quality direction
should therefore improve recovery-chain quality and stable path compression
inside the current QUBO-beam structure, rather than adding a portfolio fallback.
