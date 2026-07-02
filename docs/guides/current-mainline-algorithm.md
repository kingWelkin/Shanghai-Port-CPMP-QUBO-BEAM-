# Current Mainline Algorithm

Date: 2026-07-01

## Decision

The current promoted open120 plus paper BF/CV benchmark mainline is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish
```

This replaces the previous combined-evidence mainline:

```text
qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish
```

The promotion rule now treats open120, paper BF, and paper CV as first-class benchmark surfaces. The reason for promoting `macro_chain_witness` is not that it is the shortest possible path generator on every solved case. It is promoted because it gives the strongest current solve-coverage profile across all three surfaces while preserving the QUBO-beam architecture.

Reference roles:

```text
qubo_scheduler_state_v5_wang_rescue_polish
```

Clean historical paired baseline.

```text
qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish
```

Open120 path-quality reference. It showed that direct macro edges can keep open120 full coverage, but it is weaker on paper BF/CV than the current macro-chain line.

```text
qubo_scheduler_state_v5_qubo_diversity_union_polish
```

Previous union reference. It can be shorter on solved open120 cases, but it does not keep full open120 coverage and is much weaker on paper BF/CV.

## Promotion Evidence

Open DLTS 5x5/7x5 test-120, no business cap, `t120`, 2026-07-01:

| family | valid | solved moves |
|---|---:|---:|
| 5x5_cv1_test | 20/20 | 1388 |
| 5x5_cv2_test | 20/20 | 1303 |
| 5x5_cv3_test | 20/20 | 1160 |
| 7x5_cv1_test | 20/20 | 1046 |
| 7x5_cv2_test | 20/20 | 1004 |
| 7x5_cv3_test | 20/20 | 985 |
| all open120 | 120/120 | 6886 |

Paper BF/CV full benchmark, no business cap, `t120`, 2026-07-01:

| profile or solver | primary solved | feasibility solved | primary moves | feasible moves | full penalty | avg runtime |
|---|---:|---:|---:|---:|---:|---:|
| `macro_chain_witness_polish`, patched full aggregate | 1560/1561 | 1561/1561 | 102574 | 102637 | 102881 | 9.771 |
| `macro_chain_witness_polish`, original full run before lift repair | 1558/1561 | 1559/1561 | 102176 | 102239 | 2102483 | 9.757 |
| pure BS-FRG reproduction | 1545/1561 | 1545/1561 | 97808 | 97808 | n/a | 19.854 |
| previous `macro_credit_v2_polish` full run | 1407/1561 | 1408/1561 | 65752 | not summarized | 1100292 | 11.317 |

The patched full aggregate is the original full run with the two compact-lift error rows replaced by the targeted repair rerun. The targeted check solved all eight BF32/CV10-6 controls, including the two rows that previously failed:

| repaired row | result |
|---|---|
| `BF32/cpmp_20_8_128_52_96_2` | solved, 123 moves, 11.953s |
| `CV/10-6/data10-6-4` | solved, 275 moves, 10.089s |

The remaining known primary miss is `BF27/cpmp_20_8_96_39_58_16`, which had a feasible `N=0` path but crossed the primary time boundary in the original full run.

## Algorithm Shape

The mainline is a single beam search with QUBO selection. It is not a solver portfolio.

High-level pseudocode:

```text
solve(instance, time_limit):
    root = compact_encode(instance)
    beam = [root]
    terminal = none

    for depth in 0..max_depth:
        legal_children = []
        macro_children = []

        for parent in beam:
            legal_children += generate_compact_legal_moves(parent)

            if macro_probe_budget_allows(parent):
                chains = build_short_frg_recovery_chains(parent)
                for chain in chains:
                    if chain.has_completion_witness:
                        macro_children += lift_chain_prefix_as_candidate(parent, chain)

        candidates = legal_children + macro_children
        score each candidate with:
            immediate CPMP progress
            depth and path cost
            receiver debt
            source scarcity
            completion witness quality
            macro-chain compact path evidence

        deterministic_core = best_safe_candidates(candidates)
        qubo_pool = remaining_candidate_pool(candidates)
        selected = deterministic_core + qubo_select_quality_diversity(qubo_pool)
        selected = repair_exact_beam_cardinality(selected)

        beam = deduplicate_and_retain(selected)

        if any selected state has N == 0:
            terminal = best terminal state
            break

    if terminal is none:
        return best_partial_or_failure()

    compact_path = backtrack_selected_lineage(terminal)
    compact_path = expand_macro_chain_edges(compact_path)
    compact_path = repair_compact_successor_gaps(compact_path)
    moves = lift_compact_path_to_crane_moves(compact_path)
    moves = post_polish_valid_path(moves)
    return validator_replay(moves)
```

## What Changed From Macro Credit v2

`macro_credit_v2` mostly changed QUBO linear scoring. It rewarded macro-like candidates when they carried witness evidence and penalized raw receiver debt. That was cleaner than earlier gates, but it was still shallow: the selector saw a local state plus a credit, not a concrete short recovery chain.

`macro_chain_witness` changes the object that QUBO is allowed to select:

- A macro candidate can carry a short compact recovery-chain prefix.
- The prefix has a completion witness, so the selector sees future proof rather than a vague promise.
- Receiver debt and source scarcity are still part of the score, but they explain the chain instead of replacing it.
- The selected lineage stores enough compact-path metadata to reconstruct legal crane moves.
- Reconstruction validates successor continuity and repairs compact gaps before final replay.

This is the cleaner BS-FRG integration: BS-FRG contributes recoverable future objects; QUBO still decides which objects survive in the beam.

## QUBO Role

The QUBO selector remains narrow and interpretable.

- It does not generate crane moves.
- It does not take over the main search.
- It does not run BS-FRG after failure.
- It selects among legal one-step children and lifted legal macro-chain prefixes.
- Its pair terms preserve quality-diversity among candidate origins, source stacks, and recovery signatures.
- Its linear terms incorporate progress, path cost, receiver debt, source scarcity, and witness quality.

The important boundary is that every selected object must be replayable through the compact state transition system and final validator.

## Remaining Failure And Quality Risks

Known current risks:

- `BF27/cpmp_20_8_96_39_58_16` is the remaining full-BF/CV primary miss after the lift repair. It reached feasibility but crossed the primary time boundary.
- CV common-solved paths are still longer than pure BS-FRG on many rows. The current mainline wins coverage and runtime, not every CV path-length comparison.
- A same-command full BF/CV `t120` rerun can be done for publication hygiene, but the patched aggregate already replaces all known compact-lift error rows.

These risks should drive the next round:

1. stabilize early hits for short feasible BF paths;
2. reduce CV path length without losing coverage;
3. keep macro chains inside the QUBO selection boundary rather than adding fallback solvers.

## Operational Rule

Use this profile for current open120 and paper BF/CV benchmark work:

```text
qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish
```

Use `qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish` only as the previous combined-evidence baseline.

Use `qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish` when the question is specifically about direct macro-edge ablation or open120-only path quality.

Use `qubo_scheduler_state_v5_wang_rescue_polish` as the clean historical paired baseline.

## Reproduction

Open test-120 no-cap mainline:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish --time-limit 120 --alpha 10 --jobs 8 --no-business-cap --output docs\experiments\mainline_macro_chain_nocap_open120_t120.csv
```

Paper BF/CV full no-cap gate:

```powershell
python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF CV --variant qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish --time-limit 120 --alpha 10 --jobs 8 --no-business-cap --output docs\experiments\mainline_macro_chain_nocap_paper_bf_cv_all_t120.csv
```
