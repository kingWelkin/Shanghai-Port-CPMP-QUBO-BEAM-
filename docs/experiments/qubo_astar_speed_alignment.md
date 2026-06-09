# QUBO 与 A* 速度对齐记录

日期：2026-06-09

## 结论

若目标是向 A* 的速度对齐，当前最合适的主线不是 dynamic QUBO，而是非 dynamic 的 QUBO Beam post-polish 版本。

加速后，这个版本在全 30 个算例上仍然保证 `N=0`，总步数 509，平均时间 2.274 秒。参考 A* no-proof 为总步数 514，平均时间 2.829 秒。因此当前 QUBO Beam 速度版比 A* no-proof 快约 19.6%，同时少 5 步。

dynamic QUBO 可以作为质量增强线，但目前继续向 A* 速度压缩会付出步数代价。新固定的 `speed2000` 档能保持 30/30，但总步数为 511，比 `balanced2200` 多 3 步，速度收益不稳定。

## 主要对比

| 方法 | 全 30 解出 | 总步数 | 平均时间 | 最大时间 | 26 号位解出 | 26 号位步数 | 26 号位平均时间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A* no-proof | 30/30 | 514 | 2.829s | - | 15/15 | 419 | 5.303s |
| 非 dynamic QUBO Beam post-polish，原速度版 | 30/30 | 509 | 2.597s | 5.229s | 15/15 | 414 | 4.578s |
| 非 dynamic QUBO Beam post-polish，加速后 | 30/30 | 509 | 2.274s | 4.752s | 15/15 | 414 | 4.107s |
| dynamic QUBO `speed2000`，本次复跑 | 30/30 | 511 | 4.104s | 9.759s | 15/15 | 416 | 7.515s |
| dynamic QUBO `balanced2200` | 30/30 | 508 | 4.137s | 9.619s | 15/15 | 413 | 7.561s |
| dynamic QUBO 质量档 | 30/30 | 507 | 4.902s | 10.841s | 15/15 | 412 | 9.074s |

说明：A* 数字来自 `docs/experiments/astar_qubo_baseline_beam_comparison.md`。其余数字来自本目录内对应 CSV。

## 本次新增

### 非 dynamic 速度优化

新增结果：

```text
docs/experiments/compact_beam_qubo_speed_optimized_v2_full30_repeat.csv
```

核心优化：

```text
QUBO repair 阶段预先矩阵化 pair penalty，避免重复计算候选相似度。
compact beam 内部使用 cached successor，缓存单栈高度和单栈 N 贡献。
默认 stream_candidate_limit=0 时，主循环直接写 candidates，不再逐候选调用 stream helper。
```

这三项不改变搜索目标和约束，复跑后仍为 30/30、总步数 509。

### dynamic 速度预设

在 `scripts/benchmark_compact_beam_qubo_n0.py` 中新增 dynamic preset：

```text
--dynamic-profile speed2000
```

该 preset 等价于：

```text
dynamic_min_frontier=1200
dynamic_core_frontier=1200
dynamic_max_frontier=3200
dynamic_candidate_pool=12000
dynamic_qubo_keep_threshold=0.15
qubo_group_max_vars=96
qubo_group_select=24
qubo_group_take=4
stream_candidate_limit=2000
stream_candidate_trim_factor=2.0
```

它的定位是 dynamic QUBO 的速度探索档，不建议替代 `balanced2200` 作为主推档。

## stream 限制实验

对非 dynamic enhanced 版本额外加 `stream_candidate_limit`，没有得到更好速度：

| 配置 | 26 号位解出 | 26 号位步数 | 26 号位平均时间 |
|---|---:|---:|---:|
| stream 1200 | 11/15 | 303 | 5.639s |
| stream 1600 | 13/15 | 357 | 6.449s |
| stream 2000 | 15/15 | 417 | 7.298s |
| 原非 dynamic QUBO speed | 15/15 | 414 | 4.578s |

因此，直接给非 dynamic enhanced 加候选流式上限不是有效提速方向。

## 推荐写法

主线应写为：QUBO Beam 速度版在保持 30/30 和 `N=0` 的同时，平均时间从 2.597 秒降到 2.274 秒；相比 A* no-proof 的 2.829 秒，速度快约 19.6%，并且总步数从 514 降到 509。

dynamic QUBO 用作质量增强：`balanced2200` 为中间档，508 步、4.137 秒；质量档为 507 步、4.902 秒。`speed2000` 只作为向 A* 速度靠近的探索预设，当前不构成主推结果。
