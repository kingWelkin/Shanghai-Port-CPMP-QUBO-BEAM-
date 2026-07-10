# Hardness + Solution Quality Summary

## Definitions

- `hardness_index = 100 * (0.30 * size_norm + 0.25 * density + 0.30 * bad_ratio + 0.15 * initial_scarcity_risk)`.
- `size_norm = log1p(container_count) / max(log1p(container_count))` across the analyzed corpus.
- `solution_quality_moves_per_container = moves / container_count`; lower is better for cross-size scatter plots.
- Raw `moves` remains the primary per-instance solution-quality metric when comparing same-case before/after runs.
- `full_quality_penalty` is retained only as a feasibility/unfinished-solution guard, not the quality axis for the all-solved snapshot.
- The V4.0 row separates coverage from path quality: coverage uses the stable all-solved snapshot, while Open/Paper move totals use the polish/repaired mainline rows.
- The reference matrix reports both solved coverage and solved-case move totals; move totals are comparable only within the same benchmark scope.
- Hardness intentionally excludes runtime, moves, and final solution status.

## Repaired v4.0 Snapshot

| suite   |   cases |   solved |   moves |   full_quality_penalty |   median_hardness |   p90_hardness |   median_relocations_per_container |
|:--------|--------:|---------:|--------:|-----------------------:|------------------:|---------------:|-----------------------------------:|
| BF      |     721 |      721 |   46435 |                  46435 |           70.0168 |        79.7513 |                            0.7625  |
| CV      |     840 |      840 |   60966 |                  60966 |           76.8773 |        86.0479 |                            1.36905 |
| Open120 |     120 |      120 |    6930 |                   6930 |           79.4677 |        82.7416 |                            1.75143 |

## V4.0 Polish Quality Totals

{"cases": 1681, "solved": 1678, "moves": 113348.0, "full_quality_penalty": 114471.0, "median_hardness": 73.94152965972133, "scope": "v4.0 polish/repaired mainline rows before stable guard patch synthesis"}

## Beam-only vs. Repaired Advantage

| suite   |   both_solved_cases |   total_moves_saved |   median_moves_saved |   improved_share |   worsened_share |
|:--------|--------------------:|--------------------:|---------------------:|-----------------:|-----------------:|
| Open120 |                 120 |                1152 |                    5 |         0.883333 |        0.0333333 |
| BF      |                 719 |                 808 |                    0 |         0.337969 |        0.112656  |
| CV      |                 839 |                4320 |                    3 |         0.752086 |        0.0882002 |

## Reference Coverage Matrix

| algorithm       | Open120_rate   | Open120_label   | Open120_moves   | Open120_moves_label   | Paper BF/CV_rate   | Paper BF/CV_label   | Paper BF/CV_moves   | Paper BF/CV_moves_label   |   Real30_rate | Real30_label   |   Real30_moves | Real30_moves_label   | note                                                                                           |
|:----------------|:---------------|:----------------|:----------------|:----------------------|:-------------------|:--------------------|:--------------------|:--------------------------|--------------:|:---------------|---------------:|:---------------------|:-----------------------------------------------------------------------------------------------|
| V4.0 stable     | 1.0            | 120/120         | 6930.0          | polish moves 6930     | 1.0                | 1561/1561           | 106418.0            | polish moves 106418       |      1        | 30/30          |            509 | moves 509            | coverage uses stable all-solved; Open/Paper move totals use v4.0 polish/repaired mainline rows |
| BS-FRG pure     | 1.0            | 120/120         | 4121.0          | moves 4121            | 0.9897501601537476 | 1545/1561           | 97808.0             | moves 97808               |      1        | 30/30          |            516 | moves 516            | paper-style pure run, no QUBO mainline; Open/Paper/Real30 t20                                  |
| Wang BS-B paper | 1.0            | 120/120         | 4569.0          | moves 4569            | 0.57847533632287   | 903/1561            | 32650.0             | moves 32650               |      1        | 30/30          |            757 | moves 757            | adapted Wang-Jin-Lim BS-B baby-move beam baseline                                              |
| Classic beam    | 0.7            | 84/120          | 2711.0          | moves 2711            | 0.8392056374119155 | 1310/1561           | 62492.0             | moves 62492               |      0.933333 | 28/30          |            455 | moves 455            | no-QUBO beam; Open/Paper t120; Real30 uses budget-best historical control                      |
| A* no-proof     |                | not scalable    |                 |                       |                    | not scalable        |                     |                           |      1        | 30/30          |            514 | moves 514            | exact-search reference on 30 real single-bay cases only                                        |

## Reading

The scatter plot separates layout difficulty from solver outcome. Points drifting upward at high hardness are cases where quality pressure remains high.

The component charts split the hardness index into size, density, initial bad ratio, and scarcity risk. They are diagnostic plots, not replacement scorecards.

The advantage chart isolates the repair/polish layer on cases solved by both beam-only and repaired modes, so move savings are not inflated by feasibility differences.
