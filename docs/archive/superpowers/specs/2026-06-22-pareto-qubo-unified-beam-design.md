# Pareto QUBO Unified Beam Design

Date: 2026-06-22

## Goal

Build an ablation-friendly `scheduler_state_v5_pareto_union_polish` profile that can beat the current `scheduler_state_v5_qubo_diversity_union_polish` baseline on both benchmark families:

- Open120 must not lose solved cases against the paired same-day baseline.
- Paper BF/CV sample10 at `t20` must not drop below the current `572/611` solved result.
- If solved count is unchanged, promotion requires lower solved moves or lower full penalty.
- A one-case gain with any new failure is not promotable.

The change targets the observed failure mode where large-case beams fill with similar parents, sources, signatures, and `n` bands. The current rollout probe changed candidate ordering but did not change the beam structure enough to affect real paths.

## Non-Goals

- No new fallback solver.
- No BS-FRG or Wang engine handoff.
- No case-specific gates.
- No move-generation changes.
- No learning algorithm.
- No change to CPMP legality semantics.

## Architecture

The new profile keeps the existing layered compact beam search and uses a single unified frontier selector:

1. Rank all legal child states with the existing v5 scoring pipeline.
2. Build bounded candidate features for the top candidate window.
3. Reserve a small Pareto seed set across orthogonal objectives.
4. Fill the remaining exploration slots with one QUBO selector.
5. Fill any still-empty slots by deterministic rank order.

This replaces scattered exploration supplements with one selector surface. The deterministic core remains protected, and QUBO continues to select already-legal states only.

## Candidate Features

Each candidate receives a compact feature record:

- `quality`: current rank energy, `n`, `depth`.
- `space`: empty slots, average height pressure, max-height pressure.
- `structure`: parent code, source stack, destination stack, source/destination pair, top-group signature, reduce-like signature, `n` band.
- `repairability`: Wang pair feature score, FRG transition release proxy, bad-block release delta.
- `risk`: same-parent collapse key, same-source key, same-destination key, low-slack risk, ordered-stack disruption proxy.

Features must be cached per layer by compact state code where possible. Missing or expensive features fall back to neutral values rather than discarding candidates.

## Pareto Seed

The Pareto seed is a small deterministic reserve, not a fallback path. It chooses representatives from the same ranked candidate list:

- Best quality candidates.
- Best space/slack candidates.
- Best repairability candidates.
- Lowest risk candidates.
- Distinct source/destination pair representatives.
- Distinct parent/signature representatives.

The seed size is bounded by `pareto_union_seed_width` and counts toward the final beam. It must not force a bad candidate into the beam after the deterministic core is already full.

## QUBO Fill

After deterministic core and Pareto seed selection, the QUBO selector fills up to `pareto_union_qubo_width` remaining slots.

Linear energy rewards:

- lower normalized quality energy;
- lower `n`;
- higher repairability;
- higher slack/space;
- lower risk.

Pair penalties discourage:

- same parent;
- same source;
- same destination;
- same source/destination pair;
- same top-group signature;
- same reduce-like signature;
- same `n` band;
- same risk bucket.

The QUBO model uses the existing exact-k repair path. If the sampler returns an invalid cardinality, repair must produce exactly the requested number when enough candidates exist.

## Profile Defaults

Initial experimental defaults:

- `pareto_union_enabled=True`
- `pareto_union_candidate_window=128`
- `pareto_union_seed_width=32`
- `pareto_union_qubo_width=32`
- `pareto_union_max_vars=96`
- `pareto_union_n_band_size=2`
- `pareto_union_quality_weight=1.00`
- `pareto_union_repair_weight=0.20`
- `pareto_union_space_weight=0.12`
- `pareto_union_risk_weight=0.16`
- pair penalties start near current union-diversity values and add destination/risk penalties conservatively.

These values are intentionally broad rather than case tailored. Follow-up ablations may vary widths and weights, but the first implementation must expose them cleanly through profile fields and runner CSV telemetry.

## Data Flow

`_select_next_frontier` continues to produce ranked legal children. When the profile is active:

1. `_select_scheduler_state_qubo_frontier` dispatches protect/contract/blend/rescue as before.
2. The protect/enhanced frontier path calls the Pareto union selector before generic QUBO supplements.
3. The selector receives only the ranked candidate list and current args.
4. It returns selected codes, QUBO-selected codes, QUBO call count, QUBO var count, and QUBO runtime.
5. The outer search records telemetry in the existing result row.

No selected code can bypass compact-state deduplication or legal child generation.

## Telemetry

Add CSV fields:

- `pareto_union_enabled`
- `pareto_union_layers`
- `pareto_union_seed_selected_total`
- `pareto_union_qubo_calls`
- `pareto_union_qubo_vars`
- `pareto_union_qubo_time_sec`
- `pareto_union_qubo_selected_total`
- `pareto_union_quality_selected`
- `pareto_union_space_selected`
- `pareto_union_repair_selected`
- `pareto_union_risk_selected`
- `pareto_union_distinct_parent_count_last`
- `pareto_union_distinct_source_count_last`
- `pareto_union_distinct_dest_count_last`
- `pareto_union_distinct_signature_count_last`

These fields are required because promotion depends on proving the selector changes beam composition, not merely solver labels.

## Error Handling

- Empty candidate lists return the existing empty frontier result.
- Feature extraction errors must use neutral feature values and increment a telemetry counter.
- QUBO sampler failure falls back to repaired deterministic lowest-energy selection from the same bounded candidate window.
- Invalid parameters are clamped to safe positive ranges.
- If the Pareto seed overfills the requested beam size, trim by existing rank order.

## Testing

Focused tests:

- Pareto seed keeps candidates from distinct objective winners.
- QUBO fill penalizes same parent/source/destination/signature when quality is tied.
- Exact-k repair returns the requested number of selected states.
- Neutral feature fallback does not drop candidates.
- Open runner accepts `qubo_scheduler_state_v5_pareto_union_polish`.
- Paper runner accepts the same solver profile.

Smoke benchmarks:

- Open first 10 cases for wiring and telemetry.
- Paper sample-per-family 1 at `t5` paired with `qubo_scheduler_state_v5_qubo_diversity_union_polish`.

Decision benchmarks:

- Paper sample10 `t20` paired baseline and new profile.
- Open120 `t10` paired baseline and new profile only if the paper gate does not regress.

## Promotion Rule

Promote only if the paired run satisfies one of these:

- more solved cases on paper sample10 and no open120 solved regression;
- same solved cases on both sets and lower full penalty or solved moves on at least one set without worsening the other materially.

Otherwise keep the profile as an ablation and report the failure mode.

## Self-Review

- There are no placeholders.
- The design changes only frontier selection and benchmark wiring.
- It does not introduce fallback, case-specific gates, or new move generation.
- The benchmark gates explicitly compare against the current best union baseline.
