# A* / QUBO / Baseline Beam Comparison

> 注：本文保留 A*、QUBO Beam、classic Beam 的历史对比口径。最新主推结果是稳定版 QUBO Beam speed：30/30 达到 `N=0`，总步数 509，平均时间 2.274 秒；详见 `docs/experiments/qubo_astar_speed_alignment.md` 和 `docs/experiments/compact_beam_qubo_speed_optimized_v2_full30_repeat.md`。

Baseline beam is the strict ablation classic no-QUBO beam: compact byte-state and dominated pruning are kept, QUBO/post-polish/diversity supplements are disabled.

Per-case CSV: `docs/experiments/astar_qubo_baseline_beam_comparison.csv`

## All Cases

| algorithm | solved | avg final N | avg moves solved | total moves solved | avg runtime sec | total runtime sec | avg visited | avg peak/open | max peak/open |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A* no-proof | 30/30 | 0.000 | 17.133 | 514 | 2.829 | 84.859 | 690473 | 630139 | 2132814 |
| current QUBO enhanced | 30/30 | 0.000 | 16.967 | 509 | 3.453 | 103.589 | 17314 | 1080 | 1200 |
| dynamic QUBO | 30/30 | 0.000 | 16.900 | 507 | 4.902 | 147.045 | 33022 | 2656 | 4126 |
| baseline beam time-matched | 21/30 | 7.067 | 12.333 | 259 | 3.542 | 106.271 | 24194 | 1974 | 6400 |
| baseline beam budget-best | 28/30 | 1.467 | 16.250 | 455 | 3.177 | 95.322 | 21615 | 1534 | 3200 |

## Bay 26

| algorithm | solved | avg final N | avg moves solved | total moves solved | avg runtime sec | total runtime sec | avg visited | avg peak/open | max peak/open |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A* no-proof | 15/15 | 0.000 | 27.933 | 419 | 5.303 | 79.541 | 1343018 | 1224795 | 2132814 |
| current QUBO enhanced | 15/15 | 0.000 | 27.600 | 414 | 6.140 | 92.107 | 30875 | 1200 | 1200 |
| dynamic QUBO | 15/15 | 0.000 | 27.467 | 412 | 9.074 | 136.114 | 60508 | 3625 | 4126 |
| baseline beam time-matched | 6/15 | 14.133 | 27.333 | 164 | 6.313 | 94.694 | 42369 | 2407 | 6400 |
| baseline beam budget-best | 13/15 | 2.933 | 27.692 | 360 | 5.608 | 84.115 | 37376 | 1560 | 2400 |

## Bay 01

| algorithm | solved | avg final N | avg moves solved | total moves solved | avg runtime sec | total runtime sec | avg visited | avg peak/open | max peak/open |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A* no-proof | 15/15 | 0.000 | 6.333 | 95 | 0.355 | 5.318 | 37929 | 35483 | 140670 |
| current QUBO enhanced | 15/15 | 0.000 | 6.333 | 95 | 0.765 | 11.482 | 3753 | 960 | 1200 |
| dynamic QUBO | 15/15 | 0.000 | 6.333 | 95 | 0.729 | 10.931 | 5537 | 1686 | 3372 |
| baseline beam time-matched | 15/15 | 0.000 | 6.333 | 95 | 0.772 | 11.577 | 6019 | 1541 | 3200 |
| baseline beam budget-best | 15/15 | 0.000 | 6.333 | 95 | 0.747 | 11.208 | 5854 | 1508 | 3200 |
