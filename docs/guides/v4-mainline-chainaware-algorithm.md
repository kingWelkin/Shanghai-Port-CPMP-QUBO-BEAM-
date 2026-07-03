# v4.0 Mainline Chain-Aware QUBO Beam Algorithm

Date: 2026-07-03
Commit used for the current scorecard: `411da4b`

## Profile

The v4.0 mainline profile is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish
```

Internal solver profile:

```text
scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish
```

The profile is implemented in `scripts/benchmark_compact_beam_qubo_n0.py` and
is routed by both benchmark runners:

- `scripts/open_cpmp_benchmarks.py`
- `scripts/benchmark_compact_paper_set.py`

## Objective

The solver targets a valid `N=0` CPMP/pre-marshalling layout. It is a heuristic
search, not an exact optimality proof. The promotion scorecard currently treats
open120, paper BF, and paper CV as equally important surfaces.

The v4.0 published scorecard is no-business-cap evidence. Business-cap runs are
valid engineering checks, but they must be reported separately.

## Architectural Principle

The design rule is deliberately narrow:

> QUBO selects beam survivors; it does not generate moves, solve feasibility, or
> act as a portfolio fallback.

This gives the algorithm three safety boundaries:

1. The compact transition engine generates legal one-step moves.
2. Macro candidates carry replayable compact-path evidence.
3. Final physical moves are accepted only after validator replay.

## Main Components

### Compact Byte State

Each layout is encoded as a compact byte string. The search operates on compact
states because they are cheap to hash, compare, deduplicate, cache, and store in
lineage maps.

The state payload used by the beam is:

```text
code -> (n_value, depth, parent_code)
```

where `n_value` is the current disorder / unfinished count and `depth` is the
number of compact moves used so far.

### Legal Successor Engine

For each beam state, the solver enumerates legal top-container relocations:

```text
source stack is non-empty
receiver stack is not full
source != receiver
only the source top container moves
```

The resulting child code is scored with updated `N`, depth, and feature signals.
Dominated states are pruned by best-known depth.

### Wang Feature Signals

The v4.0 family inherits the Wang/QUBO feature lineage. Wang-style fixed-prefix
and fix-pair signals help identify locally promising states, but they remain
features. They do not take over the search and do not inject an independent
Wang solution path in the promoted profile.

### FRG Transition And Completion Signals

FRG-inspired signals describe whether a move or short chain is making useful
recovery progress:

- fill / reduce progress;
- unblocked bad-container improvement;
- receiver availability;
- source scarcity;
- max-height and pressure risks;
- completion witness quality.

These signals are used as ranking and QUBO energy terms.

### Macro-Chain Witness

The macro-chain witness supplier runs bounded BS-FRG-style completion probes
from selected beam parents. If a probe finds a useful future chain, only a short
prefix is inserted as a macro candidate.

Important boundaries:

- the full BS-FRG solution is not grafted as a fallback;
- the macro prefix enters the same candidate dictionary as ordinary children;
- QUBO decides whether that candidate survives;
- selected macro edges must later expand to compact state paths.

Key inherited parameters:

```text
macro_chain_witness_enabled = True
macro_chain_witness_parent_width = 4
macro_chain_witness_prefix_steps = 8
macro_chain_witness_max_steps = 512
macro_chain_witness_time_limit = 0.02
macro_chain_witness_min_stack_count = 6
macro_chain_witness_min_height = 8
macro_chain_witness_min_n = 12
macro_chain_witness_min_candidates = 1000
```

### Unified Macro Supply

v4.0 adds a second macro candidate supply pass. It calls the same witness
machinery, but starts from a shifted parent slice. This supplies recovery chains
that the first macro pass may miss because it focuses on the very top parents.

Key v4.0 parameters:

```text
unified_macro_supply_enabled = True
unified_macro_supply_respect_time_limit = False
unified_macro_supply_parent_width = 4
unified_macro_supply_parent_offset = 4
unified_macro_supply_max_steps = 128
unified_macro_supply_max_generated = 768
unified_macro_supply_min_candidates = 1000
```

This is the main reason v4.0 is stronger on full BF/CV: it gives the QUBO layer
more recovery candidates without adding a portfolio branch.

### Chain-Aware QUBO Selection

More macro supply can be harmful if the beam fills with near-duplicate recovery
chains. v4.0 adds macro ancestry pressure to the QUBO pair energy:

```text
union_diversity_qubo_macro_ancestry_penalty_weight = 40.0
union_diversity_qubo_macro_ancestry_penalty_scale = 4.0
frg_frontier_child_certify_time_limit = 0.004
```

This makes QUBO pay for repeatedly selecting candidates from the same macro
ancestry. The result is a better balance: enough deep recovery chains enter the
pool, but the beam does not collapse into one recovery story.

### Post-Polish

After the search finds a terminal compact path, post-polish can attempt safe
path shortening through suffix and window replacement. A replacement is accepted
only when it remains liftable, validator-valid, and strictly better under the
profile rule.

Post-polish is a quality step. It is not the source of feasibility.

## End-To-End Pseudocode

```text
run_v4(instance, time_limit):
    root = compact_encode(instance)
    beam = {root}
    best_depth[root] = 0
    parent[root] = none

    while time remains:
        candidates = {}

        for state in beam:
            for child in legal_compact_successors(state):
                if not dominated(child, best_depth):
                    candidates[child.code] = child.record

        layer = compute_layer_metrics(candidates)
        update_scheduler_state(layer)

        inject_wang_rollout_candidates_if_profile_allows(candidates)
        inject_frg_frontier_candidates_if_profile_allows(candidates)
        inject_macro_chain_witness_candidates(candidates)
        inject_unified_macro_supply_candidates(candidates)

        ranked = rank_by_n_depth_and_feature_ledger(candidates)
        core = deterministic_core(ranked)
        qubo_pool = bounded_diversity_pool(ranked - core)
        supplement = solve_union_diversity_qubo(qubo_pool)
        selected = repair_to_exact_beam_width(core + supplement)

        record_candidate_origins(selected)
        beam = selected

        if terminal_n0_exists(beam):
            terminal = best_terminal(beam)
            break

    if no terminal:
        return failure_or_best_partial()

    compact_path = backtrack_lineage(terminal)
    compact_path = expand_macro_edges(compact_path)
    compact_path = repair_compact_lift_gaps(compact_path)
    compact_path = safe_post_polish(compact_path)
    moves = lift_compact_path_to_crane_moves(compact_path)
    return validator_replay(instance, moves)
```

## Why It Works

The earlier QUBO beam variants had two complementary strengths:

- ordinary beam plus Wang features was stable and fast on local progress;
- BS-FRG-style recovery chains were good at deep low-slack recovery.

The failure mode was integration. If FRG was added as a fallback, the algorithm
became a portfolio. If FRG was added only as a shallow score, QUBO did not see
enough future proof. If too many macro candidates were injected, the beam could
lose diversity.

v4.0 resolves this by changing the object being selected:

```text
candidate = legal state + optional compact recovery-chain evidence
```

QUBO then selects among candidates that already carry future evidence, while
chain-aware pair penalties keep the selected set diverse.

## Known Trade-Offs

- v4.0 maximizes coverage and stability, not provable optimal moves.
- Some CV paths remain longer than pure BS-FRG common-solved paths.
- The macro supply pass is intentionally stronger than a tiny gate, so runtime
  must be reported with every promotion table.
- Future quality work should improve chain quality and post-polish, not add a
  final fallback solver.
