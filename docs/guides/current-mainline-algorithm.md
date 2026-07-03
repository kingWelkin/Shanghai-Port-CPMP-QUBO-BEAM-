# Current Mainline Algorithm

Date: 2026-07-03
Version: v4.0

The current promoted open120 plus paper BF/CV benchmark mainline is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish
```

This profile supersedes:

```text
qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish
```

v4.0 is promoted because the same commit rerun solved all currently tracked
promotion surfaces: open120 and the full paper BF/CV set. It keeps the QUBO-beam
architecture intact and uses BS-FRG-style chains as selectable candidate objects,
not as a fallback solver.

## Evidence Snapshot

Fresh run from commit `411da4b`, no business cap, `t120`, `jobs=8`, 2026-07-03:

| surface | solved | moves | avg runtime sec |
|---|---:|---:|---:|
| open120 DLTS | 120/120 | 6928 | 0.882 |
| paper BF/CV full1561 | 1561/1561 | 106903 | 3.933 |
| paper BF | 721/721 | 46517 | 5.786 |
| paper CV | 840/840 | 60386 | 2.343 |

Source summaries:

- `docs/experiments/chainaware_commit411da4b_open120_t120_20260703_summary.md`
- `docs/experiments/chainaware_commit411da4b_paper_bf_cv_full1561_t120_20260703_summary.md`

## Algorithm Shape

The mainline is a single compact-state beam search with QUBO survivor selection.

High-level pseudocode:

```text
solve(instance, time_limit):
    root = compact_encode(instance)
    beam = [root]
    terminal = none

    for each layer:
        candidates = {}

        for parent in beam:
            add all legal one-step compact successors

        update layer pressure signals:
            best_n, candidate pressure, parent concentration, slack,
            FRG layout pressure, source scarcity, receiver debt

        add optional evidence candidates:
            Wang rollout candidates, if enabled by profile
            FRG frontier children, if enabled by profile
            macro-chain witness candidates
            unified macro-supply candidates from shifted parent offsets

        score all candidates:
            immediate N reduction and depth
            Wang fixed-prefix / pair features
            FRG transition and completion witness quality
            receiver debt and source scarcity
            macro path quality and macro ancestry reuse

        selected = deterministic core + QUBO diversity supplement
        repair selected set to exact beam cardinality
        deduplicate and store parent lineage

        if selected contains N=0:
            terminal = best selected terminal
            break

    if terminal is none:
        return best partial failure record

    compact_path = backtrack terminal lineage
    compact_path = expand selected macro edges
    compact_path = repair compact successor gaps
    moves = lift compact path to physical crane moves
    moves = post-polish only if a strictly better valid path is found
    return validator_replay(moves)
```

## What v4.0 Adds Over Macro-Chain Witness

`macro_chain_witness` already let QUBO select short BS-FRG-style recovery-chain
prefixes with completion evidence. v4.0 keeps that base and adds a second,
controlled supply path:

- `unified_macro_supply_enabled=True`
- `unified_macro_supply_parent_width=4`
- `unified_macro_supply_parent_offset=4`
- `unified_macro_supply_max_steps=128`
- `unified_macro_supply_max_generated=768`
- `unified_macro_supply_min_candidates=1000`
- `union_diversity_qubo_macro_ancestry_penalty_weight=40.0`
- `union_diversity_qubo_macro_ancestry_penalty_scale=4.0`

The intent is to feed more deep-recovery candidates into the same QUBO selection
pool, but to penalize selecting many candidates that share the same macro
ancestry. That is the "chain-aware" part: more supply, less duplicate lineage
collapse.

## Boundaries

- QUBO does not generate crane moves.
- QUBO does not prove optimality.
- BS-FRG is not run as a final fallback.
- Macro candidates must be replayable as compact state paths.
- Physical moves are accepted only after lift and validator replay.
- The published v4.0 scorecard uses `--no-business-cap`; internal business-cap
  runs should be reported separately.

## Further Reading

- [v4.0 mainline algorithm](v4-mainline-chainaware-algorithm.md)
- [v4.0 flow diagrams](v4-mainline-flow-diagrams.md)
- [v4.0 scorecard and reproduction](v4-mainline-scorecard-and-reproduction.md)
