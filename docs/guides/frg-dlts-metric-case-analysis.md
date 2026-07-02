# FRG/DLTS Metric Case Analysis

This note maps the metric definitions from
`C:/Users/Donghao/Documents/xwechat_files/wxid_298p7qs1r88522_6ef2/temp/RWTemp/2026-06/27836c2772901586bc4014b757538896/metric_definition.md`
onto the current CPMP engineering baseline.

## Baseline

The benchmark baseline remains `qubo_scheduler_state_v5_wang_rescue_polish`:

| variant | solved | full-quality penalty | solved moves | solved gap sum | avg runtime sec |
|---|---:|---:|---:|---:|---:|
| `qubo_scheduler_state_v5_wang_rescue_polish` | 118/120 | 3873 | 3692 | 350 | 3.901 |

The metric analysis is therefore not a replacement objective. It is a gate for
where the next QUBO or FRG/DLTS-inspired supplement is allowed to intervene.

## Engineering Proxy

For single-bay `data/input/*_input.txt` cases, stacks are bottom-to-top and
`H=6`.

| metric | proxy used here |
|---|---|
| `C` | number of containers |
| `rho` | `C / (S * H)` |
| `N0` | current `future_disorder_count_stacks` value |
| `D0` | number of stacks with at least one first-inversion blocker segment |
| `B0` | sum of blocker segment depths after the first inversion in each dirty stack |
| `F0` | total empty slots, `S * H - C` |
| `L0` | number of distinct destination stacks that are safe for at least one blocker |
| `Gamma0` | average Jaccard overlap between blocker safe-landing sets |
| `BufferScarcity` | `1 - safe_landing_total / (blockers * (available_stacks - 1))` |
| `Hardness proxy` | `N0 / (L0 + 1) + rho + Gamma0` |

This proxy follows the metric definition's intent: hard cases are not just
large cases; they have many blockers, few safe landing basins, and high
competition for the same basins.

## Case Table

| case | C | rho | N0 | D0 | B0 proxy | F0 | blockers | safe landing total | L0 usable stacks | BufferScarcity | Gamma0 proxy | Hardness proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `A1_01` | 20 | 0.333 | 5 | 4 | 6 | 40 | 5 | 5 | 5 | 0.889 | 0.600 | 1.767 |
| `A1_26` | 40 | 0.667 | 24 | 10 | 52 | 20 | 24 | 41 | 8 | 0.756 | 0.377 | 3.710 |
| `A2_01` | 20 | 0.333 | 4 | 3 | 5 | 40 | 4 | 13 | 4 | 0.639 | 0.875 | 2.008 |
| `A2_26` | 40 | 0.667 | 23 | 9 | 49 | 20 | 23 | 25 | 6 | 0.845 | 0.633 | 4.586 |
| `A3_01` | 20 | 0.333 | 5 | 4 | 6 | 40 | 5 | 7 | 4 | 0.844 | 0.375 | 1.708 |
| `A3_26` | 40 | 0.667 | 25 | 9 | 56 | 20 | 25 | 64 | 7 | 0.634 | 0.362 | 4.154 |
| `C5_01` | 20 | 0.333 | 5 | 5 | 5 | 40 | 5 | 4 | 4 | 0.911 | 0.600 | 1.933 |
| `C5_26` | 40 | 0.667 | 24 | 8 | 52 | 20 | 24 | 37 | 6 | 0.807 | 0.436 | 4.531 |

## Reading

`26` bay cases are consistently harder than their `01` counterparts, but the
useful explanation is not merely size. They double `C`, halve `F0`, and push
`N0` from roughly 4-5 into 23-25. The best phrase for papers or reports is:

> `26` bay cases fall into a high-hardness region in `C`, `rho`, `N0`, `F0`,
> `L0`, and `Gamma0`; they are not just larger layouts.

`A2_26` is the cleanest example of why QUBO helps. Its `Hardness proxy` is the
highest in this sample because `N0=23`, only six destination stacks are useful
safe basins, `BufferScarcity=0.845`, and blocker safe-landing overlap is high
(`Gamma0=0.633`). A plain Top-W beam can easily retain many states that look
equally good by `N` but consume the same future basins.

`A3_26` has slightly larger `N0=25`, but it has more total safe landings and
lower conflict density. That makes it a large, hard case, but less diagnostic
for future-basin redundancy than `A2_26`.

## Algorithm Consequences

The next implementation should avoid a fixed "start QUBO at iteration 8"
story. The better rule from the metric definition is:

```text
Use QUBO iff the candidate set exceeds beam width and at least one risk signal
is high: score degeneracy, parent/path collapse, or buffer scarcity.
```

For the current v5 line, that means:

| component | current role | metric-driven extension |
|---|---|---|
| deterministic core beam | preserves the best `N, depth` candidates | remains mandatory baseline |
| Wang pair QUBO | supplements parent/fix-pair diverse candidates | keep as a light, ablation-friendly rescue |
| new FRG signals | score fill/reduce transitions and ordered-stack preservation | add only as a small candidate supplement first |
| QUBO activation | currently mostly profile/gate driven | add `Deg_t`, `Collapse_t`, and `BufferScarcity_t` gates |
| QUBO pair penalty | currently parent/similarity-oriented | move toward future-basin similarity: safe landing overlap, critical blocker overlap, and parent/action overlap |

The first safe implementation step is the independent FRG signal module. The
next risky step is not compound continuation; it is a metric-triggered
supplement that adds a few FRG/DLTS-value candidates only when the current
frontier is saturated and the future-basin risk is high.

## Cleanup Guidance

Keep:

- `qubo_scheduler_state_v5_wang_rescue_polish` as the single-route quality baseline.
- Wang pair-QUBO as a candidate supplementation mechanism.
- FRG fill/reduce signals as a deterministic scoring layer.
- DLTS lessons as policy/value telemetry and ranking, not as a trained model.

Clean up or avoid reviving:

- fixed-step QUBO start rules;
- position-overlap-only similarity;
- standalone learning routes;
- full hybrid stacks where cap, rollout, and pair-QUBO are all enabled without
  a metric trigger;
- routes that improve one smoke case but do not beat the 118/120, 3873 penalty
  baseline in a clean open-120 run.
