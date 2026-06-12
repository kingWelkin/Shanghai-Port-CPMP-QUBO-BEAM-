# QUBO 影响增强 profile 试验记录

本文记录 2026-06-12 对两条“扩大 QUBO 作用”路线的复现实验。结论先写清楚：当前主线仍是 `speed_v2`，不建议把 frontier QUBO 继续加大作为默认方向。

## 两个试验分支

### 1. `post_polish_v1`

保持 `speed_v2` 的主搜索和 frontier QUBO 参数不变，只加强主搜索归零后的后处理：

- `post_polish_passes`: `2 -> 3`
- `post_polish_suffix_max_candidates`: `48 -> 72`
- `post_polish_suffix_select`: `5 -> 7`
- `post_polish_suffix_expanded`: `60000 -> 90000`
- 其余 suffix/window 深度和候选上限小幅增加

这一路只尝试压缩已找到的 `N=0` 路径，不负责把失败路径救成 `N=0`。

### 2. `frontier_quality_v1`

保持 `speed_v2` 的 post-polish 不变，加大 frontier QUBO 的选择影响：

- `max_qubo_calls`: `32 -> 64`
- `max_qubo_vars`: `64 -> 96`
- `qubo_supplement_width`: `24 -> 32`
- `qubo_focus_n`: `8 -> 10`
- `qubo_cohorts_per_layer`: `2 -> 3`
- `qubo_parent_weight`: `0.15 -> 0.30`
- 新增 `qubo_next_n_weight=0.75`

`qubo_next_n_weight` 是一步未来 `N` 偏置：对每个 QUBO 候选状态枚举一步合法栈顶移动，估计下一步可达到的最低 `N`，并在 QUBO 线性项中惩罚未来 `N` 更差的候选。默认值为 `0.0`，所以 `speed_v2` 行为不变。

## 内部 30 算例

同机重跑，均要求 `final_n=0`。

| profile | 解出 | 总步数 | 总时间 sec | 平均时间 sec | QUBO vars | post saved |
|---|---:|---:|---:|---:|---:|---:|
| `speed_v2` rerun | 30/30 | 509 | 58.113 | 1.937 | 22554 | 1 |
| `post_polish_v1` | 30/30 | 509 | 59.239 | 1.975 | 22558 | 1 |
| `frontier_quality_v1` | 30/30 | 510 | 79.037 | 2.635 | 60624 | 1 |

内部 30 上，`post_polish_v1` 基本持平；`frontier_quality_v1` 多用约 2.7 倍 QUBO vars，却多 1 步且更慢。

## 外部 DLTS 7x5 cv2/cv3

范围：`7x5_cv2_test` 前 10 个 + `7x5_cv3_test` 前 10 个，`time_limit=10`。

| profile | 解出 | 成功算例总步数 | 总时间 sec | 成功算例平均 gap | QUBO vars |
|---|---:|---:|---:|---:|---:|
| `speed_v2` | 19/20 | 686 | 51.358 | 5.842 | 27721 |
| `post_polish_v1` | 19/20 | 685 | 62.713 | 5.789 | 27754 |
| `frontier_quality_v1` | 19/20 | 688 | 72.891 | 5.947 | 75409 |

外部 7x5 上，`post_polish_v1` 少 1 步、gap 略降，但时间更高，也没有救出失败的 `7x5_cv2_test_9`。`frontier_quality_v1` 成功率不变，步数和 gap 都变差，QUBO 变量数显著增加。

## 结论

- `speed_v2` 仍是默认主线。
- `post_polish_v1` 可以保留为轻量质量试验 profile，但当前证据只支持“可能略降步数”，不支持替代主线。
- `frontier_quality_v1` 保留为负结果复现 profile，不推荐作为后续调参主线。
- 简单扩大 frontier QUBO 变量、调用次数、补充宽度，并加入一步未来 `N` 偏置，并没有证明 QUBO 更强；反而显示当前瓶颈更可能在候选生成/搜索结构，而不是继续加大 QUBO 选择器。

## 复现命令

```powershell
python scripts\benchmark_compact_beam_qubo_n0.py --solver-profile speed_v2 --output docs\experiments\speed_v2_full30_rerun_probe.csv
python scripts\benchmark_compact_beam_qubo_n0.py --solver-profile post_polish_v1 --output docs\experiments\post_polish_quality_full30_probe.csv
python scripts\benchmark_compact_beam_qubo_n0.py --solver-profile frontier_quality_v1 --output docs\experiments\frontier_quality_full30_probe.csv
```

```powershell
python scripts\open_cpmp_benchmarks.py --families 7x5_cv2_test 7x5_cv3_test --limit-per-family 10 --variant qubo --solver-profile speed_v2 --time-limit 10 --output docs\experiments\open_cpmp_7x5_cv23_speed_v2_probe.csv
python scripts\open_cpmp_benchmarks.py --families 7x5_cv2_test 7x5_cv3_test --limit-per-family 10 --variant qubo --solver-profile post_polish_v1 --time-limit 10 --output docs\experiments\open_cpmp_7x5_cv23_post_polish_v1_probe.csv
python scripts\open_cpmp_benchmarks.py --families 7x5_cv2_test 7x5_cv3_test --limit-per-family 10 --variant qubo --solver-profile frontier_quality_v1 --time-limit 10 --output docs\experiments\open_cpmp_7x5_cv23_frontier_quality_v1_probe.csv
```
