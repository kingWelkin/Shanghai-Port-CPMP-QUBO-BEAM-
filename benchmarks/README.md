# Tracked Promotion Benchmark Corpus

This directory contains the exact benchmark inputs needed to reproduce the
v4.3 promotion gate from a normal repository clone. It intentionally excludes
the full upstream DLTS checkout, pretrained models, and exploratory datasets.

| Surface | Inputs | References | Tracked path |
|---|---:|---:|---|
| Open120 | 120 | 120 | `open120/` |
| Paper BF | 721 | - | `paper/BF/BF/` |
| Paper CV | 840 | - | `paper/CV/CRPTestcases_Caserta/` |

## Provenance

- Open120 is the first 20 naturally sorted test instances from each of
  `5x5_cv1_test`, `5x5_cv2_test`, `5x5_cv3_test`, `7x5_cv1_test`,
  `7x5_cv2_test`, and `7x5_cv3_test` in the
  [DLTS repository](https://github.com/ahottung/DLTS), source commit
  `3e6efe1c0bc40ca99d6fc0ac8e9a75157c917362`. Its upstream GPL-3.0 license is
  preserved as `open120/DLTS-LICENSE.txt`.
- BF is the complete 721-instance benchmark distributed from
  [Shunji Tanaka's pre-marshalling benchmark page](https://sites.google.com/site/shunjitanaka/pmp).
- CV is the 840-instance Caserta benchmark selected by the paper runner's
  `dataH-S-i.dat` naming rule from the
  [University of Hamburg benchmark archive](https://www.bwl.uni-hamburg.de/en/iwi/forschung/projekte/dataprojekte/brp-instances-caserta-etal-2012.zip).

The source files are copied without content transformation. `external/`
remains the location for optional full upstream clones and is intentionally
ignored by Git.

## Reproduction

```powershell
python scripts\open_cpmp_benchmarks.py --source benchmarks\open120 --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_wang_route_v43_polish --time-limit 120 --alpha 10 --jobs 8 --no-business-cap --output docs\experiments\v43_open120_t120.csv

python scripts\benchmark_compact_paper_set.py --source benchmarks\paper --datasets BF CV --sample-per-family 9999 --variant qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_wang_route_v43_polish --time-limit 120 --alpha 10 --jobs 8 --no-business-cap --output docs\experiments\v43_paper_bf_cv_full1561_t120.csv
```

The promoted result CSVs and summaries are tracked under
`docs/experiments/v43_wang_reintegration_20260709/` and
`docs/experiments/v43_full_gate_20260709/`.
