# Quality Reservoir Union Design

## Status

Approved for design on 2026-06-30. Implementation is not approved until this
spec is reviewed.

## Goal

Create an ablation profile that keeps the current compact Union + QUBO search
structure while moving the useful part of the strong FRG gate into a continuous,
quality-aware candidate reservoir.

The candidate variant name should be:

```text
qubo_scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish
```

The solver profile should be:

```text
scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish
```

The design target is not to maximize open120 alone. The promotion target treats
these benchmark groups as equally important:

- open120;
- paper BF;
- paper CV.

Any profile that improves one group while materially degrading another must stay
an experiment instead of becoming the mainline.

## Current Evidence

The latest soft-supply experiment showed that more FRG macro supply can recover
open120 feasibility:

- soft supply: open120 `120/120`, moves `4110`, penalty `4110`;
- adaptive macro: open120 `119/120`, moves `4079`, penalty `4158`;
- strong gated FRG union: open120 `120/120`, moves `3765`, penalty `3765`.

The same experiment did not close the paper BF/CV gap:

- soft supply paper smoke: primary solved `3/12`, feasible `6/12`, penalty
  `3646`;
- adaptive macro paper smoke: primary solved `3/12`, feasible `4/12`, penalty
  `3646`;
- strong gated FRG union paper smoke: primary solved `3/12`, feasible `12/12`,
  penalty `3640`.

This implies that raw candidate volume is useful but insufficient. The strong
gate appears to help because it changes the quality distribution of FRG macro
candidates, not only their count.

## Design Principle

Use a single selection mechanism:

```text
candidate quality reservoir + QUBO beam selection
```

All mechanisms are continuous scoring, filtering, or pair-energy terms. The
profile must not introduce:

- family-specific or case-specific gates;
- a BS-FRG fallback that returns an outside solution;
- a second independent beam that bypasses the Union selector;
- FRG-generated crane moves that skip the existing legality checks.

The search remains a beam search. QUBO remains a selection layer.

## Data Flow

```text
core beam children
+ Wang pair candidates
+ FRG macro/frontier candidates
        |
        v
candidate feature enrichment
        |
        v
quality-filtered reservoir
        |
        v
QUBO beam selector
        |
        v
next beam frontier
```

The reservoir sits between candidate generation and QUBO selection. It should
improve the candidate pool that QUBO sees, not replace QUBO.

## Candidate Feature Model

Each candidate receives a feature bundle. Missing values should default to
neutral scores so existing core and Wang candidates remain valid.

### Progress Features

- `n_delta`: reduction in badly placed containers relative to the parent.
- `ordered_delta`: improvement in ordered prefix or ordered sequence signal.
- `depth_cost`: number of moves consumed to create this candidate.
- `chain_efficiency`: progress per macro-chain move.

### Witness Features

- `witness_found`: whether a short completion or repair witness was found.
- `witness_final_n`: final `n` reached by the witness.
- `witness_moves`: witness path length.
- `witness_quality`: normalized score that rewards lower `witness_final_n` and
  shorter witness paths.

Witnesses are evidence for scoring only. They do not become fallback solutions.

### Debt And Scarcity Features

- `receiver_debt`: whether the candidate consumes receiver slack or creates a
  cleanup burden.
- `source_scarcity_gain`: whether the move releases scarce source capacity or a
  high-pressure blocking stack.
- `height_risk`: whether the resulting layout approaches stack cap or creates a
  fragile high receiver.
- `slack_after`: remaining aggregate slack after the move.

### Diversity Features

- `origin`: core, Wang, FRG frontier, FRG compound, or future macro origin.
- `parent_code`: parent layout identity.
- `source_signature`: source stack or source group signature.
- `receiver_signature`: receiver stack or receiver risk signature.
- `macro_signature`: compact signature for FRG chain shape.
- `n_band`: coarse `n` bucket.

## Reservoir Filter

The reservoir should be a quality filter, not a hard route switch.

Recommended first version:

1. Generate the same core, Wang, and FRG candidate families already available to
   the Union variants.
2. Enrich each candidate with the feature bundle.
3. Compute a scalar `reservoir_quality`:

```text
reservoir_quality =
    progress_reward
  + witness_reward
  + source_scarcity_reward
  - receiver_debt_penalty
  - height_risk_penalty
  - chain_length_penalty
```

4. Keep a bounded pool using both quality and diversity:
   - keep top candidates by `reservoir_quality`;
   - keep a small origin-balanced reserve;
   - keep a small high-witness reserve;
   - keep a small high-progress reserve.
5. Pass the reservoir candidates into the existing QUBO selector.

This gives FRG macro candidates a path into the beam when they are high quality,
without forcing every high-pressure layer to accept them.

## QUBO Energy Changes

The QUBO selection problem remains a beam-selection problem.

### Linear Energy

The first implementation should add continuous feature terms to the existing
candidate energy:

```text
energy(i) =
    base_cost(i)
  - a * witness_quality(i)
  - b * n_delta(i)
  - c * source_scarcity_gain(i)
  + d * receiver_debt(i)
  + e * height_risk(i)
  + f * chain_length_penalty(i)
```

The coefficients must be exposed as profile parameters so they can be ablated.

### Pair Energy

Pair penalties should discourage redundant reservoir choices:

```text
pair(i, j) penalizes:
  same parent over-concentration
  same source signature
  same receiver-debt pattern
  same n-band
  same FRG macro signature
```

Pair terms should not ban same-origin candidates. They should only make a
diverse equivalent set cheaper.

## Cross-Benchmark Promotion Rule

Promotion must use a grouped scorecard with equal benchmark groups:

```text
score = (open120_score + BF_score + CV_score) / 3
```

Each group score should consider, in order:

1. primary solved count;
2. feasible count when primary solved is not available;
3. full penalty;
4. solved moves;
5. runtime.

The profile can be promoted only if it is not materially worse than the current
best known route on any of the three groups. A single open120 win is not enough.

## Telemetry

The profile should emit enough telemetry to explain whether the reservoir is
helping or just adding noise:

- `quality_reservoir_generated_total`;
- `quality_reservoir_kept_total`;
- `quality_reservoir_kept_by_origin`;
- `quality_reservoir_selected_total`;
- `quality_reservoir_selected_by_origin`;
- `quality_reservoir_witness_found_total`;
- `quality_reservoir_avg_witness_quality`;
- `quality_reservoir_avg_receiver_debt`;
- `quality_reservoir_avg_height_risk`;
- `quality_reservoir_pair_penalty_total`;
- `quality_reservoir_on_path_selected_total`.

Open120 and paper benchmark CSV rows should expose these fields.

## Test Plan

Focused unit tests:

- feature enrichment defaults are neutral for candidates without witness data;
- reservoir keeps high-quality candidates and preserves bounded origin balance;
- reservoir does not behave like a family or case gate;
- QUBO linear energy rewards witness quality and progress;
- QUBO pair energy penalizes duplicate parent/source/receiver signatures;
- open and paper runners accept the new variant;
- CSV row serialization includes the new telemetry.

Smoke benchmarks:

- open120 focused residual cases;
- BF/CV known hard subset;
- one mixed smoke that includes at least one open family, one BF family, and one
  CV family.

Decision benchmarks:

- full open120;
- full BF;
- full CV;
- paired comparison against strong gated and the current Union mainline.

## Implementation Boundaries

The first version should be small enough to ablate:

- add feature enrichment helpers;
- add reservoir scoring/filtering helpers;
- connect the reservoir before QUBO selection;
- add profile parameters and telemetry;
- add runner wiring and tests.

It should not refactor the whole solver or change baseline behavior.

## Risks

The main risk is building a softer-looking hard gate. To avoid that, every
decision must be parameterized as a score, threshold, quota, or pair penalty
that can be ablated and measured.

The second risk is candidate bloat. The reservoir must have a bounded size and
must report how many candidates it generated, kept, and selected.

The third risk is overfitting to open120. The equal-weight BF/CV/open120
scorecard is part of the design, not an afterthought.

## Review Checklist

- No case-specific or family-specific rule is required.
- The profile remains opt-in.
- QUBO remains the selector.
- FRG macro candidates do not bypass legality checks.
- Promotion requires open120, BF, and CV evidence.
- All new constants are ablation-friendly profile parameters.
