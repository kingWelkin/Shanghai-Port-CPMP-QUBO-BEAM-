# Current QUBO Beam Algorithm Details

Date: 2026-07-03
Version: v4.0

This note describes the current QUBO beam mechanism behind:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish
```

For the complete v4.0 architecture and flow diagrams, see:

- [v4.0 mainline algorithm](v4-mainline-chainaware-algorithm.md)
- [v4.0 flow diagrams](v4-mainline-flow-diagrams.md)

## QUBO Boundary

QUBO never creates a crane move. It receives already legal or replayable compact
candidate states and decides which candidates should survive into the next beam.

The candidate set can contain:

- ordinary legal one-step compact children;
- Wang-feature-enriched children;
- FRG frontier children;
- macro-chain witness prefixes;
- unified macro-supply prefixes.

Every selected candidate must still pass compact lineage reconstruction, physical
move lifting, and validator replay.

## Selection Objective

The union diversity QUBO is a small selector over a bounded candidate window.
Each binary variable means "keep candidate `i` in the supplement".

Conceptually:

```text
minimize
    sum_i quality_energy_i * x_i
  + sum_{i<j} similarity_penalty_ij * x_i * x_j
  + exact_k_penalty(sum_i x_i, target)
```

The linear energy favors:

- lower `N`;
- shallower depth;
- Wang fixed-prefix / pair quality;
- FRG transition quality;
- completion witness quality;
- source scarcity relief;
- receiver-debt control;
- macro path evidence.

The pair energy penalizes:

- same parent over-concentration;
- same source-stack families;
- similar reduction signatures;
- same `N` band crowding;
- repeated macro origins;
- repeated macro ancestry in v4.0.

## Deterministic Core Plus QUBO Supplement

The selector is intentionally not all-QUBO. Each layer keeps a deterministic
core first, then uses QUBO for a small diversity supplement. This keeps the
search stable while allowing QUBO to preserve candidates that a simple sorted
beam would discard.

```text
ranked = sort candidates by (N, depth, feature ledger, parent, code)
core = first safe deterministic candidates
qubo_pool = bounded remaining candidates
qubo_selected = solve small diversity QUBO
next_beam = exact_width_repair(core + qubo_selected)
```

## Why Chain-Aware Matters

Earlier macro profiles could inject recovery chains, but several selected
candidates often came from the same macro ancestry. That made the beam look
diverse by state code while still exploring the same recovery story.

v4.0 adds:

```text
union_diversity_qubo_macro_ancestry_penalty_weight = 40.0
union_diversity_qubo_macro_ancestry_penalty_scale = 4.0
```

This does not ban macro candidates. It asks QUBO to pay for selecting many
macro candidates that share the same ancestor chain, which makes the extra
unified macro supply safer.

## Post-Polish

After a terminal compact path is found, post-polish may try to shorten it. The
current line accepts a polish only when the replacement is valid and strictly
better under the configured acceptance rule.

Post-polish is not a rescue fallback:

- it does not turn an unsolved `N>0` path into a published solution;
- it does not bypass compact lifting;
- it does not bypass validator replay.

## Telemetry To Watch

Useful columns in benchmark CSVs:

- `qubo_calls`, `qubo_vars`, `qubo_selected_total`;
- `macro_chain_witness_calls`, `macro_chain_witness_added_total`,
  `macro_chain_witness_selected_total`;
- `unified_macro_supply_calls`, `unified_macro_supply_added_total`,
  `unified_macro_supply_selected_total`;
- `post_polish_attempts`, `post_polish_improvements`,
  `post_polish_moves_saved`;
- `compact_lift_repair_inserted`, `frg_terminal_witness_used`.

When a future profile improves results, these fields should identify whether the
gain came from supply, selection, reconstruction, or polish.
