# Open CPMP 时间匹配消融与逐 case 拆解

本轮目标是回答一个更严格的问题：外部 DLTS 算例上，QUBO 的少步数是否只是因为它比 classic no-QUBO 花了更多时间。

做法是用同一 case 的 QUBO 实际运行时间作为 classic no-QUBO 的时间预算，并扫 classic beam width `1200/2400/4800`。每个 case 取 classic 的最好结果作为 `classic_time_matched_best`。

## 口径

- QUBO reference 来自已有 `open_cpmp_benchmarks.py` 结果。
- classic no-QUBO 禁用 frontier QUBO 和 post-polish QUBO，`selection_policy=classic`。
- `classic_time_matched_best` 是同一 case、同一时间预算下三个 classic width 的最好结果。
- `cv1` 使用唯一箱号语义；`cv2/cv3` 使用 priority-group 语义。
- 所有成功行仍要求 `final_n=0`、只移动栈顶箱、不能移出堆场，并通过独立验解。

## 5x5 结果

| family | variant | cases | N=0 solved | moves total | avg runtime sec | avg budget sec | avg ref gap |
|---|---|---:|---:|---:|---:|---:|---:|
| 5x5_cv1_test | QUBO Beam | 20 | 10 | 281 | 1.067 | 1.067 | 2.600 |
| 5x5_cv1_test | classic time-matched best | 20 | 12 | 360 | 0.697 | 1.067 | 4.500 |
| 5x5_cv2_test | QUBO Beam | 20 | 14 | 368 | 1.205 | 1.205 | 2.000 |
| 5x5_cv2_test | classic time-matched best | 20 | 14 | 385 | 0.797 | 1.205 | 3.071 |
| 5x5_cv3_test | QUBO Beam | 20 | 18 | 460 | 1.153 | 1.153 | 2.056 |
| 5x5_cv3_test | classic time-matched best | 20 | 20 | 518 | 0.729 | 1.153 | 2.000 |

5x5 的结论是混合的：classic 在 `cv1/cv3` 成功率更高，但成功解质量通常更差；`cv2` 两者同为 14/20，QUBO 少 17 步，平均 gap 更低。

## 7x5 结果

| family | variant | cases | N=0 solved | moves total | avg runtime sec | avg budget sec | avg ref gap |
|---|---|---:|---:|---:|---:|---:|---:|
| 7x5_cv1_test | QUBO Beam | 10 | 4 | 149 | 2.728 | 2.728 | 5.750 |
| 7x5_cv1_test | classic time-matched best | 10 | 6 | 249 | 1.981 | 2.728 | 10.500 |
| 7x5_cv2_test | QUBO Beam | 10 | 9 | 330 | 3.252 | 3.252 | 6.222 |
| 7x5_cv2_test | classic time-matched best | 10 | 9 | 349 | 2.185 | 3.252 | 8.333 |
| 7x5_cv3_test | QUBO Beam | 10 | 10 | 356 | 3.214 | 3.214 | 5.500 |
| 7x5_cv3_test | classic time-matched best | 10 | 10 | 366 | 1.853 | 3.214 | 6.500 |

7x5 的质量信号更清楚：`cv2/cv3` 在同成功率下，QUBO 分别少 19 步和 10 步，平均 gap 也更低。`cv1` 中 classic 多解 2 例，但 QUBO 成功解质量明显更好。

## 逐 case 分类

| family | qubo only | classic only | both QUBO fewer | both classic fewer | both tie | neither |
|---|---:|---:|---:|---:|---:|---:|
| 5x5_cv1_test | 2 | 4 | 2 | 2 | 4 | 6 |
| 5x5_cv2_test | 2 | 2 | 3 | 2 | 7 | 4 |
| 5x5_cv3_test | 0 | 2 | 2 | 3 | 13 | 0 |
| 7x5_cv1_test | 0 | 2 | 4 | 0 | 0 | 4 |
| 7x5_cv2_test | 0 | 0 | 5 | 1 | 3 | 1 |
| 7x5_cv3_test | 0 | 0 | 6 | 1 | 3 | 0 |

7x5 `cv2/cv3` 是当前最支持 QUBO 质量优势的分支：两族合计 20 例中，QUBO 少步 11 例，classic 少步 2 例，7 例持平或无解差异。

## case 列表

5x5:

- `5x5_cv1_test` QUBO only：`5x5_cv1_test_3`, `5x5_cv1_test_13`
- `5x5_cv1_test` classic only：`5x5_cv1_test_8`, `5x5_cv1_test_10`, `5x5_cv1_test_18`, `5x5_cv1_test_19`
- `5x5_cv2_test` QUBO only：`5x5_cv2_test_3`, `5x5_cv2_test_9`
- `5x5_cv2_test` classic only：`5x5_cv2_test_6`, `5x5_cv2_test_17`
- `5x5_cv3_test` classic only：`5x5_cv3_test_2`, `5x5_cv3_test_12`

7x5:

- `7x5_cv1_test` classic only：`7x5_cv1_test_3`, `7x5_cv1_test_6`
- `7x5_cv2_test` QUBO fewer：`7x5_cv2_test_1`, `7x5_cv2_test_2`, `7x5_cv2_test_4`, `7x5_cv2_test_5`, `7x5_cv2_test_7`
- `7x5_cv3_test` QUBO fewer：`7x5_cv3_test_1`, `7x5_cv3_test_2`, `7x5_cv3_test_4`, `7x5_cv3_test_6`, `7x5_cv3_test_7`, `7x5_cv3_test_9`

## QUBO 贡献拆解

贡献标记来自结果行中的 `post_polish_moves_saved`、`qubo_selected_on_path`、`qubo_ancestor_on_path` 和 `qubo_selected_total`：

- `frontier_selected` 表示该 case 的搜索过程中 QUBO 确实选入过补充状态，但不等于该状态必然在最终路径上。
- `frontier_direct` 表示 QUBO 直接选入的状态进入最终路径。
- `frontier_ancestor` 表示最终路径上存在 QUBO 选入状态的后代。
- `post_polish` 表示后处理 QUBO 有实际步数压缩。

观察：

- 5x5 的 QUBO 少步 case 中，`post_polish` 更常见，说明这部分质量收益有一部分来自后处理压缩。
- 7x5 的 QUBO 少步 case 中，`frontier_selected` 普遍存在，但 `frontier_direct/ancestor` 较少；更合理的说法是 QUBO 稀疏改变搜索前沿，而不是持续主导最终路径。
- 7x5 `cv2/cv3` 中 post-polish 只覆盖少数 QUBO 少步 case，因此这里的质量优势不能完全归因于 post-polish。
- adaptive QUBO 在 7x5 上更快，但会牺牲部分成功率或少量质量，仍作为速度/质量变体，不替代 `speed_v2` 主线。

## 关于 beam width

本轮再次确认，classic beam 不是 width 越大越好。大量 case 中 `w2400/w4800` 在相同时间预算下反而更差，因为更宽的 frontier 提高了每层候选生成和排序负担，导致有效搜索深度下降。当前 time-matched best 经常来自 `w1200`。

## 当前推荐写法

> 在外部 DLTS 时间匹配消融中，QUBO Beam 的优势不应表述为所有分支上的成功率优势。更准确的结论是：在重复优先级和 7x5 分支上，QUBO 在同成功率下显著改善成功解质量；在 5x5 分支上，classic time-matched 有时成功率更高，但平均 gap 和步数质量不稳定。QUBO 的贡献主要体现为稀疏前沿纠偏和少量后处理压缩。

## 复现命令

```powershell
python scripts\open_cpmp_time_matched_ablation.py --reference-csv docs\experiments\open_cpmp_dlts_5x5_cv1_200cases_both.csv docs\experiments\open_cpmp_dlts_5x5_cv23_test20_priority_groups_all.csv --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test --max-cases-per-family 20 --widths 1200 2400 4800 --output docs\experiments\open_cpmp_time_matched_5x5_cv123.csv
```

```powershell
python scripts\open_cpmp_time_matched_ablation.py --reference-csv docs\experiments\open_cpmp_dlts_7x5_cv1_40cases_both.csv docs\experiments\open_cpmp_dlts_7x5_cv23_test10_priority_groups_all.csv --families 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --max-cases-per-family 10 --widths 1200 2400 4800 --output docs\experiments\open_cpmp_time_matched_7x5_cv123.csv
```

明细输出：

- `open_cpmp_time_matched_5x5_cv123_summary.md`
- `open_cpmp_time_matched_7x5_cv123_summary.md`
- `open_cpmp_time_matched_5x5_cv123_breakdown.csv`
- `open_cpmp_time_matched_7x5_cv123_breakdown.csv`
