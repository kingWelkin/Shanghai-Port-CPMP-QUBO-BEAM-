# QUBO Beam 与 A* 速度对齐结论

日期：2026-06-09

## 主结论

当前推荐主线是非 dynamic 的 QUBO Beam post-polish 速度版。它在 30 个单贝算例上全部达到 `N=0`，总步数 509，平均运行时间 2.274 秒。

相比 A* no-proof，QUBO Beam 速度版同时做到：

- 解出率相同：30/30；
- 步数更少：509 对 514，少 5 步；
- 速度更快：2.274 秒对 2.829 秒，平均快约 19.6%；
- 26 号位更稳：4.107 秒对 5.303 秒，快约 22.5%。

因此，文稿里建议强调：QUBO 模块不是只带来质量提升，也能在工程优化后达到更好的速度-质量组合。

## 对比结果

| 方法 | 解出 | 总步数 | 平均时间 | 最大时间 | 26 号位解出 | 26 号位步数 | 26 号位平均时间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A* no-proof | 30/30 | 514 | 2.829s | 17.121s | 15/15 | 419 | 5.303s |
| 原 QUBO Beam speed | 30/30 | 509 | 2.597s | 5.229s | 15/15 | 414 | 4.578s |
| 稳定版 QUBO Beam speed | 30/30 | 509 | 2.274s | 4.752s | 15/15 | 414 | 4.107s |
| dynamic QUBO `balanced2200` | 30/30 | 508 | 4.137s | 9.619s | 15/15 | 413 | 7.561s |
| dynamic QUBO 质量档 | 30/30 | 507 | 4.902s | 10.841s | 15/15 | 412 | 9.074s |

数据来源：

- A* no-proof：`docs/experiments/astar_qubo_baseline_beam_comparison.md`
- 稳定版 QUBO Beam：`docs/experiments/compact_beam_qubo_speed_optimized_v2_full30_repeat.csv`

## 稳定版改动

稳定版没有改变问题约束、目标函数或可行动作集合。仍然要求：

- 只能移动栈顶箱；
- 不允许把箱子移出栈；
- 目标为 `N=0`；
- 每条输出路径通过业务约束和目标校验。

主要加速点是工程层面的重复计算消除：

- QUBO repair 阶段预先矩阵化 pair penalty，避免重复计算候选相似度；
- compact beam 内部使用 cached successor，缓存单栈高度和单栈 `N` 贡献；
- 默认 `stream_candidate_limit=0` 时直接写入 candidates，避免每个候选额外经过 stream helper。

## 复现命令

```powershell
python scripts\benchmark_compact_beam_qubo_n0.py --output docs\experiments\compact_beam_qubo_speed_optimized_v2_full30_repeat.csv
```

验证命令：

```powershell
$env:PYTHONPATH='.;src;scripts'
pytest -q tests\test_qubo.py tests\test_astar.py
python -m compileall -q container_reshuffle scripts tests
```

## 写法建议

推荐表述：

> 稳定版 QUBO Beam 在 30 个算例上全部达到 `N=0`，总步数为 509，低于 A* no-proof 的 514；平均时间为 2.274 秒，快于 A* no-proof 的 2.829 秒。结果表明，QUBO 模块在保留 Beam 搜索速度优势的同时，能引导搜索获得更短解。

dynamic QUBO 建议作为质量增强变体单独报告。它可以把总步数进一步降到 508 或 507，但运行时间高于速度版，不应作为速度对齐主线。
