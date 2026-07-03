# Current Best Algorithm Note

Date: 2026-07-03
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
