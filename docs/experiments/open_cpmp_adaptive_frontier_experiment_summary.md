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
| 上港 30 例 | QUBO Beam `speed_v2` | 30 | 30 | 2.237 | - |
| 上港 30 例 | adaptive_qubo | 30 | 27 | 1.748 | - |

## 判断

`adaptive_qubo` 对 7x5 的作用明确：成功数从 QUBO Beam 的 13/40 提升到 22/40，且平均时间从 2.431 秒降到 2.014 秒。代价是已解实例的参考 gap 从 4.231 变差到 6.955，因此它是更偏成功率/速度的 7x5 外部 profile，不是质量最优 profile。

5x5 上原 QUBO 已优于 classic，因此开放 runner 对 5x5 保持原 QUBO 宽度，不再使用未保护的 800 frontier 结果。

上港 30 例不能切到 adaptive：它只解出 27/30。主线仍是固定 `speed_v2` QUBO Beam。

## 复现命令

```powershell
python scripts\open_cpmp_benchmarks.py --source $env:TEMP\dlts_cpmp_source --families 7x5_cv1_test 7x5_cv1_validation --limit-per-family 20 --variant qubo_adaptive --time-limit 3 --alpha 10 --output docs\experiments\open_cpmp_dlts_7x5_cv1_40cases_adaptive.csv
```

```powershell
python scripts\benchmark_compact_beam_qubo_n0.py --profile speed_v2 --selection-policy adaptive_qubo --output docs\experiments\adaptive_qubo_speed_v2_full30.csv
```
