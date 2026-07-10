# v4.3 Mainline Algorithm

Date: 2026-07-10
Version: v4.3

## Profile

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_wang_route_v43_polish
```

v4.3 promotes a deterministic pre-solve policy over the strongest validated
routes from the v4 family. It does not run two solvers and choose afterward.

## Routing Policy

```text
solve_v43(instance, source):
    metrics = initial_scarcity_metrics(instance)
    duplication = priority_duplication_ratio(instance)

    if source is DLTS
       and stack_count in {5, 7}
       and max_height == 7
       and container_count in {25, 35}:
        profile = wang_rescue_successor_cache_polish
    else if layout shape matches the small route
            and light_route_score(metrics) >= threshold
            and duplication >= 0.45:
        profile = wang_rescue_successor_cache_polish
    else:
        profile = chainaware_short_incumbent_v42_polish

    result = compact_qubo_beam(instance, profile)
    return validator_replay(result.moves)
```

The source and priority-duplication checks matter because DLTS and paper CV can
share the same stack/height/container shape while using different priority
semantics. Shape alone is not a safe route discriminator.

## Wang Short Route

The short route uses the v5 Wang rescue profile with pair-QUBO and successor
cache polishing. It is preferred on the small open-style layouts where its
local fixed-prefix and receiver ordering signals consistently produce much
shorter paths than the deep chain-aware route.

## Chain-Aware General Route

The general route keeps the v4.0 architecture:

1. Generate legal compact one-step children.
2. Add Wang features and BS-FRG-inspired macro-chain witnesses.
3. Add unified macro supply from shifted parent offsets.
4. Let QUBO select a quality-diverse beam while penalizing repeated ancestry.
5. Expand macro edges and repair compact transition gaps.
6. Lift the compact path to legal crane moves.
7. Run bounded BS-FRG short-incumbent compression only when it returns a shorter
   validator-clean path.
8. Accept only after final validator replay.

QUBO selects beam survivors; it does not generate crane moves or prove
optimality. BS-FRG supplies and compresses recovery chains; it does not bypass
the validator.

## Full Evidence

| Surface | Solved | Moves | Average runtime |
|---|---:|---:|---:|
| Open120 | 120/120 | 3,759 | 6.252s |
| BF | 721/721 | 45,906 | 6.512s |
| CV | 840/840 | 55,865 | 3.081s |
| Open120 + BF + CV | 1,681/1,681 | 105,530 | - |
| Real30 | 30/30 | 525 | 2.446s |

## Reproduction

```powershell
python scripts\open_cpmp_benchmarks.py --source benchmarks\open120 --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_wang_route_v43_polish --time-limit 120 --alpha 10 --jobs 8 --no-business-cap --output docs\experiments\v43_open120_t120.csv

python scripts\benchmark_compact_paper_set.py --source benchmarks\paper --datasets BF CV --sample-per-family 9999 --variant qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_wang_route_v43_polish --time-limit 120 --alpha 10 --jobs 8 --no-business-cap --output docs\experiments\v43_paper_bf_cv_t120.csv
```

The repository-tracked corpus and its provenance are documented in
[`benchmarks/README.md`](../../benchmarks/README.md).

## Evidence Files

- `docs/experiments/v43_wang_reintegration_20260709/v43_wang_route_open120_t120_all_dlts_j8_summary.md`
- `docs/experiments/v43_full_gate_20260709/v43_wang_route_paper_bf_cv_t120_j8_summary.md`
- `docs/experiments/v43_full_gate_20260709/v43_real30_mainline_branch_t120_summary.md`
