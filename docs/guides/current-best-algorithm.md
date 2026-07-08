# Current Best Algorithm Note

Date: 2026-07-07
Version: v4.0

## Short Answer

For the equal-weighted open120 plus full paper BF/CV promotion gate, the current
best and recommended mainline is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish
```

The previous promoted line is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish
```

The clean historical paired baseline remains:

```text
qubo_scheduler_state_v5_wang_rescue_polish
```

The opt-in quality mode for long-path CV compression is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_quality_polish
```

It keeps the v4.0 main search unchanged and only adds BS-FRG post-compress
witness replacement on long paths. Use it when path length matters more than
runtime; keep the promoted mainline above for default fast full-surface runs.

A faster quality mode is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_quality_fast_polish
```

It uses the same post-compress mechanism with a shorter budget, suffix-first
ordering, and early accept once the path is short enough or the saved ratio is
large enough.

The experimental beam-native quality profile is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_beam_quality_polish
```

It keeps BS-FRG post-compress disabled and instead gives the beam/QUBO selector
multiple FRG completion-witness alternatives per macro parent. It is not a
promoted default until paired open120 plus full BF/CV runs show no solved-count
loss and lower move totals than v4.0.

The experimental v4.1 beam-feedback candidate is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_feedback_v41_polish
```

It also keeps BS-FRG post-compress disabled, but instead of adding another
fallback or portfolio it changes the beam/QUBO feedback signal itself: same
parent alternatives are compared locally, shorter completion witnesses receive
credit only when there is a comparable sibling, receiver debt and height risk
reduce credit, and one-layer delayed survival credit rewards signatures whose
selected parents generate progress in the next layer. The current candidate
only applies high-confidence positive feedback inside the QUBO candidate window;
negative feedback is not turned into a global penalty. It is not promoted until
the full paired open120, BF, and CV gate beats v4.0 without solved-count loss.

The first full gate check did not pass. On 2026-07-07, v4.1 solved open120
`120/120` with `7065` moves, but solved paper BF/CV only `1558/1561`
(`BF 720/721`, `CV 838/840`). Since v4.0 remains `120/120` and `1561/1561`,
v4.1 stays experimental.

## Positioning Decision

The project position is now fixed as follows:

- v4.0 is the default mainline for fast, stable, full-surface solving.
- BS-FRG post-compress quality profiles are accepted as the practical path
  compression route, especially for long CV solutions.
- Beam-native witness quality remains a research direction, not the main
  promotion target, because preserving the future-best deep recovery chain
  inside a pruned beam is much harder than validating and replacing a suffix
  after a feasible path is already available.

This means future path-length work should not treat post-compress as a failed
fallback. It is a deliberate quality mode: start from a valid v4.0 path, ask a
deep recovery-chain oracle for shorter validated segments, and only accept
improvements. Beam-native experiments are still useful, but they must be judged
against the same full open120, BF, and CV gate before becoming default.

## Why v4.0 Is Promoted

v4.0 is promoted because it has fresh full-surface evidence from the same commit:

| surface | solved | moves | avg runtime sec |
|---|---:|---:|---:|
| open120 DLTS | 120/120 | 6928 | 0.882 |
| paper BF/CV full1561 | 1561/1561 | 106903 | 3.933 |
| paper BF | 721/721 | 46517 | 5.786 |
| paper CV | 840/840 | 60386 | 2.343 |

Earlier macro-chain witness evidence was strong, but it relied on a patched
aggregate for BF/CV. v4.0 has a direct full1561 rerun with no unsolved rows.

## Interpretation

The result is a feasibility and robustness win first. Pure BS-FRG can still be
shorter on some common-solved CV rows, so future work should target path length
without breaking the single QUBO-beam architecture.

Promotion should remain strict:

- open120 must stay at 120/120;
- BF and CV must be reported separately;
- the full paper set must use `--sample-per-family 9999`;
- no profile should be promoted from smoke or residual-only evidence;
- path quality improvements must not introduce new unsolved rows.

## See Also

- [v4.0 mainline algorithm](v4-mainline-chainaware-algorithm.md)
- [v4.0 scorecard and reproduction](v4-mainline-scorecard-and-reproduction.md)
- [Algorithm profile status](algorithm-profile-status.md)
