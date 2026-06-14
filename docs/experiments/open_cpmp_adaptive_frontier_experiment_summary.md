# Adaptive Frontier 外部算例实验

## 目的

7x5 DLTS 算例比 5x5 分支数更大，固定 1200 frontier 的 QUBO Beam 在 3 秒预算下推进深度不足。这里新增 `adaptive_qubo` 选择策略：

- `best_n > 16`：frontier 收窄到 800，优先加速下潜。
- `8 < best_n <= 16`：frontier 取 1000。
- `best_n <= 8`：frontier 恢复到 1200，让末段 QUBO 保持质量。
- `adaptive_candidate_pool` 保留为开关，但默认 0。实测 stream/硬候选池裁剪会明显降低深度，不作为默认策略。
- open runner 对 `stack_count <= 5` 做保护：`qubo_adaptive` 在 5x5 上保持 1200/1200/1200，等价于原 QUBO 路径，不牺牲已有 5x5 结果。

## 关键结果

| scope | variant | cases | N=0 solved | avg runtime sec | avg ref gap among solved |
|---|---|---:|---:|---:|---:|
| DLTS 5x5_cv1 | QUBO Beam | 200 | 101 | 1.212 | 2.149 |
| DLTS 5x5_cv1 | classic no-QUBO | 200 | 84 | 0.766 | 3.000 |
| DLTS 5x5_cv1 | qubo_adaptive | 200 | 使用 5x5 保护，等价 QUBO | - | - |
| DLTS 7x5_cv1 | QUBO Beam | 40 | 13 | 2.431 | 4.231 |
| DLTS 7x5_cv1 | classic no-QUBO | 40 | 15 | 2.236 | 6.733 |
| DLTS 7x5_cv1 | qubo_adaptive | 40 | 22 | 2.014 | 6.955 |
| DLTS 7x5_cv1 | qubo_adaptive_polish | 40 | 22 | 2.029 | 6.864 |
| DLTS 7x5_cv1 | qubo_adaptive_rescue late | 40 | 22 | 2.021 | 6.909 |
| 上港 30 例 | QUBO Beam `speed_v2` | 30 | 30 | 2.237 | - |
| 上港 30 例 | adaptive_qubo | 30 | 27 | 1.748 | - |

## 结构感知 adaptive controller 复测

新增 `adaptive_controller_v1`，仍使用 `selection_policy=adaptive_qubo`，但不再只按 `best_n` 三段式选宽度。新规则读取层级信号：

- 候选压力高且 `N` 仍高时收窄到 800/1000；
- `N` 接近末段或候选压力不高时保持较宽 frontier；
- plateau 且 depth slack 紧时触发保护；
- 保护强度按实例结构决定：5/7 栈只做轻保护，8 栈以上允许更强的 plateau 扩宽。

本轮本地复测只用于判断新规则方向，不替代上面的历史表：

| scope | variant | cases | N=0 solved | solved moves | avg runtime sec | avg ref gap among solved |
|---|---|---:|---:|---:|---:|---:|
| 上港 30 例 | old `adaptive_qubo` | 30 | 27 | 424 | 1.768 | - |
| 上港 30 例 | `adaptive_controller_v1` | 30 | 27 | 424 | 1.857 | - |
| DLTS 5x5_cv1 test+validation | old `qubo_adaptive` | 40 | 19 | 538 | 1.100 | 2.421 |
| DLTS 5x5_cv1 test+validation | `qubo_adaptive_controller` | 40 | 19 | 538 | 1.095 | 2.421 |
| DLTS 7x5_cv1 test+validation | old `qubo_adaptive` | 40 | 19 | 705 | 2.181 | 6.474 |
| DLTS 7x5_cv1 test+validation | `qubo_adaptive_controller` | 40 | 21 | 793 | 2.100 | 6.857 |
| DLTS 7x5_cv2+cv3 test | old `qubo_adaptive` | 20 | 18 | 632 | 2.241 | 5.222 |
| DLTS 7x5_cv2+cv3 test | `qubo_adaptive_controller` | 20 | 18 | 631 | 1.939 | 5.167 |

判断：结构感知 controller 对开放 7x5 更有价值。它在 7x5 cv1 多解 2 个 case，在 cv2/cv3 保持解率并降低平均时间；5x5 不变。内部 30 只持平且略慢，因此不能替代内部主线 `speed_v2`，也不应写成通用最优 profile。

## 判断

`adaptive_qubo` 对 7x5 的作用明确：成功数从 QUBO Beam 的 13/40 提升到 22/40，且平均时间从 2.431 秒降到 2.014 秒。代价是已解实例的参考 gap 从 4.231 变差到 6.955，因此它是更偏成功率/速度的 7x5 外部 profile，不是质量最优 profile。

本轮强化试验显示，直接把 `light_rescue_qubo_v1` 接到 adaptive 早期会误伤已解 case：早救援版只有 20/40。调成 late rescue 后恢复到 22/40，并把总步数从 836 降到 835；但它没有救出新 case。更稳的增强是 `adaptive_polish_v1`：保持 22/40，把总步数降到 834，平均 gap 降到 6.864，平均时间小幅升到 2.029 秒。因此当前推荐把 `qubo_adaptive_polish` 作为 7x5 cv1 的质量增强变体，把 `qubo_adaptive_rescue` 作为负结果/后续调参入口。

5x5 上原 QUBO 已优于 classic，因此开放 runner 对 5x5 保持原 QUBO 宽度，不再使用未保护的 800 frontier 结果。

上港 30 例不能切到 adaptive：它只解出 27/30。主线仍是固定 `speed_v2` QUBO Beam。

## 复现命令

```powershell
python scripts\open_cpmp_benchmarks.py --source $env:TEMP\dlts_cpmp_source --families 7x5_cv1_test 7x5_cv1_validation --limit-per-family 20 --variant qubo_adaptive --time-limit 3 --alpha 10 --output docs\experiments\open_cpmp_dlts_7x5_cv1_40cases_adaptive.csv
python scripts\open_cpmp_benchmarks.py --source $env:TEMP\dlts_cpmp_source --families 7x5_cv1_test 7x5_cv1_validation --limit-per-family 20 --variant qubo_adaptive_polish --time-limit 3 --alpha 10 --output docs\experiments\open_cpmp_dlts_7x5_cv1_40cases_adaptive_polish.csv
```

```powershell
python scripts\benchmark_compact_beam_qubo_n0.py --profile speed_v2 --selection-policy adaptive_qubo --output docs\experiments\adaptive_qubo_speed_v2_full30.csv
```
