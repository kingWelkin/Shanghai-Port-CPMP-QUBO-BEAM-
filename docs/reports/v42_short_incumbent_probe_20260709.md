# v4.2 Short-Incumbent Probe - 2026-07-09

## Profile

`scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_short_incumbent_v42_polish`

Built on v4.0 chain-aware beam+QUBO. The main beam search stays unchanged. After a valid beam result is found, BS-FRG is invoked only as a shorter incumbent producer:

- trigger: `moves >= 120`
- BS-FRG postcompress budget: `4.0s`
- incumbent bound: enabled
- accept rule: legal replay, `final_n=0`, strictly fewer moves
- no BS-FRG portfolio replacement as default

## Fixes Made During Probe

- Added incumbent-bound plumbing to `BsFrgSolver` through `SolverConfig.exact_incumbent_moves`.
- Added `exact_stop_on_incumbent_improvement` support for future faster variants.
- Added runner variant for Open and Paper benchmark runners.
- Added Open row telemetry for `bs_frg_postcompress_*` fields.
- Fixed a gate bug: `bs_frg_postcompress_min_ratio=0.0` no longer bypasses `bs_frg_postcompress_min_moves`.

## Focused Probe Results

### Open hard sample, t20

| variant | cases | solved | moves | runtime sec | postcompress attempts |
|---|---:|---:|---:|---:|---:|
| v4.0 chain-aware | 12 | 12 | 398 | 48.94 | 0 |
| v4.2 short-incumbent | 12 | 12 | 398 | 41.24 | 0 |

Interpretation: Open small cases are already short enough, so v4.2 correctly avoids postcompress work.

### Paper hard sample, t20/no-cap

Sample: `CV/10-6` first 12 plus `BF14/BF16/BF30/BF32` first 12 each, 60 cases total.

| variant | cases | solved | solved moves | runtime sec | attempts | accepted | telemetry saved |
|---|---:|---:|---:|---:|---:|---:|---:|
| v4.0 chain-aware | 60 | 58 | 10433 | 686.87 | 0 | 0 | 0 |
| v4.2 short-incumbent | 60 | 60 | 9824 | 756.82 | 55 | 21 | 2059 |

Paired on the 58 cases solved by both: `10433 -> 9566`, saving `867` moves. v4.2 also solved two extra t20 BF cases in this probe.

By family, paired move delta:

| dataset | family | delta moves | new solved |
|---|---|---:|---:|
| CV | 10-6 | -914 | 0 |
| BF | BF14 | -52 | 1 |
| BF | BF16 | +107 | 1 |
| BF | BF30 | -43 | 0 |
| BF | BF32 | +35 | 0 |

## Main Interpretation

v4.2 is useful, but not yet a clean promotion over v4.0. The large gain is concentrated in CV long paths. BF remains noisy: some cases improve, but BF16/BF32 can regress in move count under t20. The postcompress accept rule itself is safe; regressions mostly come from budget-sensitive base-path variance and cases where BS-FRG times out without improving.

## Next Gate

Do not update current-best until full paired verification passes:

- Open120 t120/no-cap: must keep 120/120.
- BF full t120/no-cap: must keep 721/721.
- CV full t120/no-cap: must keep 840/840.
- Combined moves must improve versus same-day v4.0.

If BF regressions persist at t120, keep v4.2 as a CV/quality candidate rather than default mainline.
