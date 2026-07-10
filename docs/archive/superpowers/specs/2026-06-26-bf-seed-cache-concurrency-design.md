# BF Seed Cache Concurrency Isolation Design

## Status

Approved for specification on 2026-06-26. Awaiting user review before an implementation plan is written.

## Goal

Make BF hard-family feasibility diagnostics comparable between serial and parallel paper benchmark runs by isolating FRG seed generation from worker scheduling effects.

The first target is the strict hard-family union sample at `alpha=10`, `time_limit=20`, and families `BF14`, `BF16`, `BF30`, `BF32`, `10-6`, and `10-10`. This design does not promote any algorithm to the paper mainline. It creates the next experiment track for BF seed/cache diagnosis after the strict scorecard work.

## Evidence

The strict hard-family union baseline produced the same primary result under serial and parallel execution, but different feasibility diagnostics:

| run | cases | primary solved | feasibility | over cap | over time | full penalty |
|---|---:|---:|---:|---:|---:|---:|
| union jobs1 | 60 | 0 | 29 | 26 | 3 | 25491 |
| union jobs8 | 60 | 0 | 25 | 22 | 3 | 25491 |

The four lost jobs8 feasibility rows are all BF cases:

| family | case | jobs1 final_n | jobs1 moves | jobs1 time_ok | jobs8 final_n |
|---|---|---:|---:|---|---:|
| BF14 | `cpmp_16_8_103_21_78_5` | 0 | 1268 | false | 78 |
| BF32 | `cpmp_20_8_128_52_96_1` | 0 | 627 | false | 96 |
| BF32 | `cpmp_20_8_128_52_96_3` | 0 | 747 | false | 96 |
| BF32 | `cpmp_20_8_128_52_96_9` | 0 | 1043 | false | 96 |

None of these rows is a strict primary success because the serial `N=0` paths are over the 20-second budget. They still matter because they expose a concurrency-sensitive recovery mechanism.

The active union profile inherits `scheduler_state_v5_frg_seed_qubo_polish`, which enables `frg_seed_enabled`, sets a small seed budget, and accepts large-layout over-cap seeds as feasibility evidence. The current `_frg_seed_incumbent(...)` computes its rollout inside the solver call using the remaining per-case deadline and a local empty cache. In a process-parallel benchmark, this makes the seed result sensitive to worker contention even when the instance, profile, and nominal case limit are the same.

## Decision

Use a per-case FRG seed materialization step for BF diagnosis. Generate seed evidence once per `(dataset, family, case, profile, alpha, seed-configuration)` key under an explicit isolated budget, then let serial and parallel benchmark runs consume the same seed artifact.

This is a diagnostic experiment path, not a default solver behavior change.

## Scope

In scope:

- Add an opt-in seed-cache or seed-ledger experiment path for FRG seed materialization.
- Keep default solver behavior unchanged when no explicit seed artifact is requested.
- Make the seed artifact case-scoped and configuration-scoped.
- Record seed status, source, runtime, move count, final N, cap status, and cache hit/miss in benchmark rows or companion metadata.
- Run a targeted BF replay on the four known concurrency-sensitive cases before any wider benchmark.
- Compare `jobs=1` and `jobs=8` on identical materialized seed input.

Out of scope:

- No CV `10-6` or `10-10` receiver-debt or macro-clearing work in this milestone.
- No promotion of lazy terminal-escape portfolio.
- No mainline profile promotion.
- No attempt to count over-cap or over-time `N=0` as strict primary success.
- No full 611-case run until the targeted BF evidence is stable.

## Design

### Seed Artifact

The seed artifact should contain enough information to replay or reject a seed without relying on process timing:

| field | Purpose |
|---|---|
| `dataset` / `family` / `case` | case identity |
| `profile` | solver profile or open-runner variant identity |
| `alpha` / `time_limit` | benchmark contract identity |
| `frg_seed_time_limit` and step controls | seed configuration identity |
| `moves` or a stable move-path encoding | replayable seed body |
| `final_n` / `generated` / `runtime_sec` / `status` | diagnostic telemetry |
| `cap_ok` / `large_layout_ok` | strict-scorecard classification support |
| `cache_key` / `cache_source` | traceability for row comparison |

The implementation plan may choose JSONL, CSV plus move-path sidecar, or a compact pickle-free text format. The important rule is that the artifact is portable across worker processes and does not depend on an in-memory `lru_cache` or per-process mutable cache.

### Consumption

When a seed artifact is supplied, `run_compact_beam_instance(...)` should use it as the FRG seed candidate before falling back to live `_frg_seed_incumbent(...)`, but only under an explicit experiment flag. A cache miss must be visible in the output and must not silently change the benchmark contract.

The strict primary predicate remains unchanged:

```text
primary_ok = feasibility_ok && business_ok && time_limit_ok
```

Materialized BF seeds can improve feasibility coverage or explain why a live parallel worker lost a recovery path. They do not improve the strict primary score unless the resulting path is also cap-valid and in-time.

### Job Comparison

The first comparison should be deliberately small:

1. Materialize seed artifacts for the four known BF rows.
2. Run those four cases with `jobs=1` and artifact consumption enabled.
3. Run the same four cases with `jobs=8` and artifact consumption enabled.
4. Compare `final_n`, `moves`, `frg_seed_used`, `frg_seed_status`, `frg_seed_cap_ok`, and strict fields.

Only after this passes should the experiment expand to `BF14` and `BF32` sample10, then to the six-family sample10 strict baseline.

## Alternatives Considered

### Increase Live FRG Seed Budget

Increasing `frg_seed_time_limit` could reduce parallel misses, but it hides the actual problem: serial and parallel rows would still be consuming different live seed results. This is useful as a sensitivity probe, not as the next controlled experiment.

### Promote Lazy Portfolio

The lazy terminal-escape portfolio is not suitable for this step. It was interrupted before producing a summary artifact, and it changes the rescue surface instead of isolating the BF seed/cache mechanism. Keep it as an experiment branch.

### Start CV Macro Clearing Now

CV `10-6` and `10-10` likely need a different mechanism around receiver debt and large-scale clearing. Mixing that with BF seed/cache work would blur the evidence. CV should be opened as a separate design after BF seed isolation is measured.

## Validation

Automated tests must cover:

1. Default behavior remains unchanged when no seed artifact is provided.
2. A valid materialized seed can be consumed and reported as `frg_seed_used`.
3. An artifact with stale configuration or wrong case identity is rejected or reported as a miss.
4. A parallel runner path can read the same artifact as the serial path.
5. Strict scorecard fields continue to classify over-cap or over-time materialized seeds as primary-invalid.

Experiment validation:

1. Target the four known BF rows first.
2. Publish a paired jobs1/jobs8 comparison with identical materialized seed input.
3. If jobs8 still diverges, record whether the difference occurs after seed consumption, not inside seed generation.
4. If jobs8 aligns with jobs1, expand to BF14/BF32 sample10.
5. Only then rerun the six-family strict sample10 comparison.

## Promotion Rule

This design is complete when it can answer one question with evidence:

Can BF jobs8 consume the same seed evidence as jobs1 and reproduce the same feasibility diagnostics without changing the strict primary scorecard?

If yes, the next implementation may use the seed artifact as a controlled experiment tool. If no, the next diagnosis should move downstream from seed generation to path consumption, pruning, or post-seed beam interaction.
