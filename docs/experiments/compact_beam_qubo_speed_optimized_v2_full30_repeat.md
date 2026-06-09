# 稳定版 QUBO Beam Speed 结果

日期：2026-06-09

这是当前主推的非 dynamic QUBO Beam speed 版本。算法使用 compact byte-state 分层 Beam、受保护 QUBO 补充和后处理 polish，目标严格要求 `N=0`。

结果文件：

- 明细 CSV：`docs/experiments/compact_beam_qubo_speed_optimized_v2_full30_repeat.csv`
- 接受性 CSV：`docs/experiments/compact_beam_qubo_speed_optimized_v2_full30_repeat_acceptance.csv`
- 动作目录：`docs/experiments/compact_beam_qubo_speed_optimized_v2_full30_repeat_moves/`

## 核心结果

| 指标 | 数值 |
|---|---:|
| 算例数 | 30 |
| `N=0` 解出 | 30/30 |
| 总步数 | 509 |
| 平均运行时间 | 2.274 秒 |
| 最大运行时间 | 4.752 秒 |
| 26 号位解出 | 15/15 |
| 26 号位总步数 | 414 |
| 26 号位平均运行时间 | 4.107 秒 |

相对 A* no-proof 参考结果：

- 解出率相同：30/30；
- 总步数更少：509 对 514；
- 平均时间更低：2.274 秒对 2.829 秒；
- 26 号位平均时间更低：4.107 秒对 5.303 秒。

## 约束说明

该结果没有放宽问题约束：

- 只允许移动栈顶箱；
- 不允许将箱子移出栈；
- 接受条件必须为 `N=0`；
- 输出动作通过业务可行性和目标校验。

## 复现命令

```powershell
python scripts\benchmark_compact_beam_qubo_n0.py --output docs\experiments\compact_beam_qubo_speed_optimized_v2_full30_repeat.csv
```

## 定位

该版本适合作为论文或报告里的 QUBO Beam 主算法。dynamic QUBO 可以作为质量增强变体单独报告，因为它能把总步数降到 508 或 507，但运行时间更高。
