# Deterministic QUBO Sampler Design

## Goal

Reduce quality drift in the current compact CPMP v5 Wang/QUBO benchmark by allowing QUBO sampling to run a fixed number of reads and iterations instead of stopping on a tiny wall-clock deadline.

## Scope

This change is an opt-in experiment. Existing callers keep the current time-limited simulated annealing behavior unless they explicitly request deterministic sampling.

## Approach

- Add an opt-in `respect_time_limit` flag to `QuboSamplerConfig`.
- When `respect_time_limit=False`, `SimulatedAnnealingQuboSampler` runs all configured `num_reads * sa_iterations` steps and ignores its internal sampler deadline.
- Add a compact benchmark CLI/profile flag, `--deterministic-qubo-sampler`, that passes `respect_time_limit=False` to Wang pair-QUBO, unified Wang-QUBO, group/rescue/generic QUBO, and post-polish QUBO calls.
- Surface the flag in result rows so benchmark CSVs show whether a run used fixed-iteration QUBO sampling.

## Non-Goals

- Do not change beam ranking, Wang feature scoring, or post-polish acceptance logic.
- Do not make deterministic sampling the default for every solver.
- Do not alter external/API QUBO sampler behavior beyond fallback compatibility.

## Verification

- Add a sampler unit test proving fixed-iteration mode performs all reads even with a near-zero time limit.
- Add compact runner tests proving the CLI flag is accepted and serialized.
- Run the Wang/QUBO test subset and then a 20-second open120 experiment for `qubo_scheduler_state_v5_wang_rescue_polish`.
