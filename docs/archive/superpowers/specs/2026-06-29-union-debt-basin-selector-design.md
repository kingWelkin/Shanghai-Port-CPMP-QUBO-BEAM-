# Union Debt/Basin Selector Design

## Status

Approved for design on 2026-06-29. Implementation is not approved until this
spec is reviewed.

## Goal

Improve the current compact mainline without replacing it:

```text
qubo_scheduler_state_v5_qubo_diversity_union_polish
```

The new ablation profile is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish
```

Its solver profile should be:

```text
scheduler_state_v5_qubo_diversity_union_debt_basin_polish
```

The target is to recover some paper BF/CV hard cases while preserving the
current open120 behavior. The design must not become a BS-FRG fallback,
portfolio, or family-specific rule set.

## Evidence

The 2026-06-29 hard-case trace report split the remaining BF/CV failures into
four mechanisms:

- BF14/BF32: compact union does not reproduce short BS-FRG recovery. Successful
  paths often come from long FRG seed paths instead of the compact beam.
- CV 5-8 / 6-6: micro rollout can turn a good compact route into a longer
  feasible route by consuming receiver slack and creating cleanup debt.
- CV 10-6: search reaches low `n`, but the retained lineage falls into a dead
  basin.
- CV 10-10: the retained frontier barely lowers `n`; candidate count is large,
  but useful long-horizon recovery candidates are missing or not retained.

This argues against tuning a single QUBO coefficient. The next profile should
change what the selector regards as a good frontier, not add another fallback.

## Design Principle

Use one selector mechanism:

```text
quality + receiver debt + escape potential + basin diversity
```

All new signals affect candidate ranking or QUBO selection only. They do not
generate final crane moves by themselves, do not return an outside solution, and
do not override feasibility checks.

## Candidate Signals

### Receiver Debt

`receiver_debt` estimates whether a child consumed scarce receiver capacity
without buying enough progress.

Initial components:

- reduction in safe receiver count;
- increase in average height pressure;
- last-buffer consumption on an ordered or nearly ordered receiver;
- moving into a stack that becomes height-critical;
- no immediate `n` drop or only a weak `n` drop.

The score should be continuous, not a hard gate. A candidate with a strong `n`
drop can still survive even if it incurs debt. A candidate with no progress and
high debt should pay a linear penalty.

### Escape Potential

`escape_potential` estimates whether a child opens future reduction space.

Initial components:

- increase in safe receiver count;
- creation of a legal receiver for a currently blocked high-priority group;
- exposing a non-increasing top-down bad sequence that can be reduced soon;
- reducing stack height pressure while preserving ordered stacks;
- lowering parent-lineage congestion around the same source stack.

This is a linear bonus. It should be bounded so it cannot dominate actual
progress or business-quality terms.

### Basin Label

`basin_label` is a coarse route class used for pairwise diversity. It is not a
case label and must not include instance names.

Initial label fields:

- `n_band`;
- blocked-stack-count bucket;
- safe-receiver-count bucket;
- top-group signature bucket;
- source-stack or reduced-stack bucket;
- height-pressure bucket.

The QUBO pair term penalizes selecting two candidates with the same or very
similar basin label. This targets CV 10-6 style failures where many low-`n`
states are variants of the same dead recovery plan.

## Optional Candidate Supplement

The profile may add a small "reduction-chain seeded child" supplement, but only
inside the ordinary candidate pool.

Rules:

- generate from the same ranked parent frontier;
- cap parent count and child count tightly;
- use short FRG-style reduction chains only as child states;
- merge generated children before normal union/QUBO selection;
- keep beam width unchanged;
- mark origin for telemetry only;
- never return the chain as a fallback solution.

This addresses BF and CV 10-10 candidate-universe failures while preserving the
single selector model.

## QUBO Integration

For candidate `i`, add bounded linear adjustments:

```text
linear_i =
    existing_union_linear_i
  + w_receiver_debt * receiver_debt_i
  - w_escape_potential * escape_potential_i
  + w_chain_cost * reduction_chain_depth_i
```

For candidates `i, j`, add bounded pair adjustments:

```text
pair_ij =
    existing_pair_ij
  + w_basin_similarity * basin_similarity(i, j)
```

The cardinality repair and existing exact-k selection behavior remain
unchanged. If QUBO is disabled for a layer, the same linear score should still
be visible to the non-QUBO ranker so the profile has one coherent preference
model.

## Telemetry

Add result-level fields:

- `debt_basin_enabled`;
- `receiver_debt_selected_total`;
- `receiver_debt_avg_selected`;
- `escape_potential_selected_total`;
- `escape_potential_avg_selected`;
- `basin_pair_penalty_total`;
- `basin_unique_selected_total`;
- `reduction_chain_generated_total`;
- `reduction_chain_selected_total`;
- `debt_basin_selected_on_path`;

Add frontier trace fields:

- `receiver_debt`;
- `escape_potential`;
- `basin_label`;
- `basin_similarity_group`;
- `reduction_chain_origin`;
- `reduction_chain_depth`;
- `selected_by_debt_basin_qubo`.

The trace must make it possible to separate three outcomes:

- useful child was never generated;
- useful child was generated but not selected;
- useful child was selected but entered a dead lineage.

## Tests

Focused tests should cover:

- receiver-debt score penalizes slack-consuming non-progress moves;
- escape-potential score rewards future reduction space without changing move
  legality;
- basin similarity pair term selects different basins when quality is near tied;
- reduction-chain children enter the same candidate pool and respect beam width;
- the profile is opt-in and the current union profile remains unchanged;
- frontier trace rows expose debt, escape, basin, and chain-origin fields.

## Experiment Plan

### Step 1: Seven-Case Trace Gate

Run the exact 2026-06-29 hard/regression set:

- BF14 `cpmp_16_8_103_21_78_1`;
- BF32 `cpmp_20_8_128_52_96_3`;
- CV 6-6 `data6-6-8`;
- CV 5-8 `data5-8-1`;
- CV 5-8 `data5-8-10`;
- CV 10-6 `data10-6-7`;
- CV 10-10 `data10-10-1`.

Compare against current union and micro-tiebreak traces. The profile is not
promotable if it improves only one hard case while worsening the CV 5-8 / 6-6
compact routes.

### Step 2: Paper Sample Gate

Run paper BF/CV sample10 at 20 seconds with the same alpha and job settings as
the current scorecard. Require no solved-count regression and no full-penalty
regression before broader testing.

### Step 3: Open120 Guard

Run open120 at 10 seconds. The profile must keep current mainline quality:

- no valid-count regression below the accepted union baseline;
- no material full-penalty regression;
- no material solved-move regression.

### Step 4: Paired Decision Run

Only if the first three gates pass, run a same-day paired union versus
debt-basin comparison and produce a per-case diff. Promotion requires the paired
run, not a single hard-case win.

## Non-Goals

- No BS-FRG terminal fallback.
- No lazy portfolio between union, wide, bridge, or paper-pure BS-FRG.
- No family-specific or instance-specific gates.
- No learning algorithm.
- No beam-width increase as the main explanation for improvement.
- No default-profile replacement without paired open120 and paper BF/CV
  evidence.

## Promotion Rule

Keep `qubo_scheduler_state_v5_qubo_diversity_union_polish` as the mainline until
the new profile clears both benchmark families:

- open120: preserve the accepted union solve count and quality envelope;
- paper BF/CV: improve solved count or full penalty without damaging open120.

If the profile only improves trace diagnostics, keep it as a research ablation
and use its trace output to design the next smaller experiment.
