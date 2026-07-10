# v4.0 Path Length Trace Report

> [!IMPORTANT]
> Historical report. The promoted mainline is [v4.3](../../guides/v43-mainline-algorithm.md), validated at `1681/1681` with `105530` moves.

Date: 2026-07-06

## Conclusion

This trace pass did not find a safe promotion candidate. No new profile should be promoted over the current v4.0 chainaware baseline from these experiments.

The old open120 long-path examples are not reproducible on clean v4.0 now:

| case | clean v4 moves |
|---|---:|
| 5x5_cv1_test_12 | 35 |
| 5x5_cv2_test_1 | 30 |
| 5x5_cv2_test_10 | 29 |
| 5x5_cv2_test_12 | 31 |

The reproducible long-path issue is concentrated in larger CV cases such as CV/10-6. The selected path is dominated by expanded FRG macro witness chains.

## CV/10-6 Baseline Trace

| case | moves | macro expanded steps | macro chain selected | unified macro selected | post-polish saved |
|---|---:|---:|---:|---:|---:|
| data10-6-1 | 427 | 423 | 9 | 5 | 0 |
| data10-6-2 | 410 | 436 | 0 | 0 | 30 |
| data10-6-3 | 332 | 328 | 2 | 0 | 0 |
| data10-6-4 | 440 | 436 | 2 | 1 | 0 |
| data10-6-5 | 352 | 350 | 0 | 0 | 0 |

Root cause:

- Final moves are almost identical to `frg_macro_edge_expanded_steps_total` in most traced cases.
- The current search is robust because it accepts deep FRG macro recovery chains.
- The same mechanism creates long paths because the recovery chain itself is long.
- Local penalties on long chains can destroy solvability; several attempts made CV worse or caused a timeout.
- Goal-only sorting is too late because ancestor macro choices have already committed the path to a long recovery basin.

## Attempted Fixes

| attempt | CV/10-6 first5 result | open top4 result | decision |
|---|---:|---|---|
| candidate effective depth, weight 1.0 | 2041 moves | not promoted | reject |
| candidate effective depth, weight 0.25 | 1939 moves | not enough safety evidence | reject |
| candidate 0.25 + goal witness sort | 1950 moves | failed 5x5_cv2_test_10 once | reject |
| goal-only witness sort | 2103 moves | safe | reject |
| full-path goal witness sort | 2141 moves | safe | reject |
| macro witness move penalty weight 5 | 2087 moves | not promoted | reject |
| macro witness move penalty weight 10 | data10-6-1 failed at 120s | unsafe | reject |
| macro witness move penalty weight 20 | 2036 moves | safe on top4 | reject |
| heavier post-polish on data10-6-1 | 517 vs 427 baseline | not useful | reject |
| macro witness beam completion | open top4 safe, CV/10-6 hung | too slow | reject in current form |

## Code State

The failed named profiles were not left in runner variant lists. The current branch only keeps disabled-by-default diagnostic infrastructure:

- macro witness effective-depth scoring helper;
- full compact-path macro witness depth accounting helper;
- optional macro-chain beam witness flags, disabled by default and not exposed as a promoted variant.

## Next Direction

The next promising route is to improve macro witness generation quality before the candidate enters QUBO:

1. Generate a small set of alternative recovery prefixes per selected parent instead of one deterministic FRG chain.
2. Score those prefixes by final N, witness moves, receiver debt, source scarcity, and completion quality.
3. Cache per-parent FRG child expansions; naive BS-FRG beam calls are too expensive when repeated per layer.
4. Let QUBO choose among alternative macro prefixes, rather than applying a global long-chain penalty.
5. Promote only after paired open120 + BF + CV validation.
