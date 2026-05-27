# Greedy Gap Analysis

数据来源：

- `docs/experiments/greedy_vs_mip.csv`
- `docs/experiments/greedy_vs_mip_larger_real.csv`

这里的 MIP 最优值优先采用 `MIP-3D`。当前所有 `MIP-3D` 可解算例均已证明最优；`MIP-4D` 在可证明最优的算例上与 `MIP-3D` 目标值一致。

## 口径

正式 optimality gap 只在 Greedy 产出完整良序解时计算：

```text
absolute_gap = greedy_moves - mip_opt
relative_gap = absolute_gap / mip_opt
```

若 `mip_opt = 0` 且 `greedy_moves = 0`，记 relative gap 为 `0%`。

对于 Greedy 为 `NOT_SORTED` 的算例，Greedy 不是完整可行 pre-marshalling 解，不能作为 MIP 的上界，因此不计算正式 gap。下表中的 `shortfall_to_opt` 只是诊断量：

```text
shortfall_to_opt = mip_opt - greedy_moves
```

它表示 Greedy 当前只走了多少步，相比从初始状态完成良序所需的最优步数还差多少步；它不是 optimality gap。

## 汇总

- 总算例数：13
- Greedy 得到完整良序解：3
- Greedy 完整良序解的正式 gap：全部为 `0`，即 `0%`
- Greedy 未良序：10
- 未良序算例的 `shortfall_to_opt` 范围：`1..4` 步
- 未良序算例的平均 `shortfall_to_opt`：`2.5` 步

## 明细

| Case | Greedy status | Greedy moves | MIP-3D opt | MIP-4D opt | Formal abs gap | Formal rel gap | shortfall_to_opt |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sorted_2x2` | SORTED | 0 | 0 | 0 | 0 | 0% | 0 |
| `one_move_2x2` | SORTED | 1 | 1 | 1 | 0 | 0% | 0 |
| `two_bad_stacks_3x2` | SORTED | 2 | 2 | 2 | 0 | 0% | 0 |
| `buffer_needed_3x3` | NOT_SORTED | 1 | 4 | 4 | - | - | 3 |
| `mixed_4x3` | NOT_SORTED | 2 | 3 | 3 | - | - | 1 |
| `real_A1_stacks_1_2_3` | NOT_SORTED | 0 | 2 | 2 | - | - | 2 |
| `real_A1_stacks_1_2_4` | NOT_SORTED | 0 | 4 | 4 | - | - | 4 |
| `real_A2_stacks_1_2_8` | NOT_SORTED | 1 | 4 | 4 | - | - | 3 |
| `real_B1_stacks_1_2_3` | NOT_SORTED | 2 | 3 | 3 | - | - | 1 |
| `real_C1_stacks_1_2_3` | NOT_SORTED | 0 | 2 | 2 | - | - | 2 |
| `larger_real_A1_1_2_4_7` | NOT_SORTED | 0 | 3 | 3 | - | - | 3 |
| `larger_real_A1_1_2_3_6` | NOT_SORTED | 1 | 3 | 3 | - | - | 2 |
| `larger_real_A1_1_2_4_6_n9` | NOT_SORTED | 1 | 5 | TIME_LIMIT | - | - | 4 |

## 结论

当前算例中，Greedy 只有在最小的 3 个算例上与两个 MIP 模型达到一致最优解，gap 为 `0`。

其他 10 个算例中，Greedy 终止时尚未达到全场良序，因此没有正式 optimality gap。若只看“已执行步数距离初始最优完成步数”的诊断缺口，当前缺口为 `1..4` 步，平均 `2.5` 步。
