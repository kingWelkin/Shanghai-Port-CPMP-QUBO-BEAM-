# QUBO 影响增强 profile 试验记录

本文记录 2026-06-12 对“扩大 QUBO 作用”路线的复现实验。结论先写清楚：当前上港 30 算例主线仍是 `speed_v2`，不建议把 frontier QUBO 全局放大作为默认方向；触发式 `light_rescue_qubo_v1` / `scheduler_v1` 可作为开放 7x5 硬例的解出率增强路线。

## 试验分支

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

### 3. `rescue_qubo_v1`

这一路不是全局扩大 QUBO，而是在风险层触发：

- 最近 2 层 `best_n` plateau；
- `depth + best_n` 距业务 cap 的 slack 不超过 26；
- 当前层候选数至少 5000；
- 每例最多 16 个 rescue 层；
- rescue 层先用 `beam_width=1350/core_width=1215`，若已经触发 2 层仍未突破，则升到 `1600/1440`；
- rescue 层临时放开 `qubo_focus_n=-1`，使用 `max_qubo_vars=96`、`qubo_supplement_width=48`、`qubo_cohorts_per_layer=4`。

它针对的是 `7x5_cv2_test_9` 这类 `N` 长平台硬例：普通 `speed_v2` 在该例从未触发 frontier QUBO，因为失败路径一直停在 `N=16`，高于默认 `qubo_focus_n=8`。

### 4. `qubo_pair_policy`

这是诊断接口，不是新主线。当前支持：

```text
baseline
none
parent_only
byte_similarity
stack_signature
hybrid
```

默认 `baseline` 完全保留旧 pairwise penalty。`none` 用于证明 pairwise 项是否必要；`parent_only` 单独测试同父惩罚；`stack_signature` 和 `hybrid` 会加入 compact 栈结构相似度惩罚。

单例 `A2_26` smoke 中，使用增强 QUBO 参数时各 policy 都能合法归零，均为 29 步；`stack_signature` 和 `hybrid` 更慢，暂未显示质量收益。因此这一路目前价值主要是消融诊断，而不是直接提升结果。

### 5. `qubo_bucket`

`qubo_bucket` 让 QUBO 在启发式近似等价的桶内选择候选，而不是只在 deterministic core 之后补尾部。当前支持：

```text
--selection-policy qubo_bucket
--bucket-key n_depth|f_n|n_band
--bucket-core-ratio
--bucket-qubo-share
--bucket-max-qubo-vars
--bucket-qubo-select
```

单例 `A2_26` 上，`qubo_bucket` 能保持 29 步并降低 QUBO 调用数：`enhanced light` 为 61 次 QUBO、约 6.086 秒；`qubo_bucket n_depth` 为 22 次、约 4.806 秒；`n_band` 为 13 次、约 5.129 秒。说明这一路更像速度型 QUBO 选择变体。

外部 `7x5_cv2_test` 前 10 个上，`qubo_bucket` 为 9/10，成功算例总步数 348，总时间 27.478 秒，QUBO vars 22807；硬例 `7x5_cv2_test_9` 停在 `final_n=28`。因此它不能替代 `rescue_qubo_v1` 的硬例解出能力。

### 6. No-regret portfolio

新增 `scripts/benchmark_qubo_no_regret_portfolio_n0.py`。它先跑稳定基线，再跑 QUBO 挑战者，只选择验证通过且步数更短的结果；挑战者失败或退化时回退 `stable`。

内部 30 上，`stable + qubo_bucket` portfolio 全部归零：baseline 总步数 509，portfolio selected 总步数 508；只有 `C5_26_single` 真正少 1 步，另有 3 个算例因同步数但 bucket 更快而选择 bucket。代价是总组合时间 115.057 秒，平均每例 3.835 秒，不能与单 profile runtime 直接比较。

### 7. Protected group QUBO

现有 `group_qubo_supplement` 可作为 protected group QUBO 试验入口。按 `protected_group_medium` 参数 smoke `A2_26`：28 步、5.724 秒、72 次 QUBO、group QUBO 14 次、group vars 1740。它与 `speed_v2` stable 同步数，但更慢；当前只作为诊断路线保留。

### 8. `window_polish_v2`

`window_polish_v2` 把 QUBO 放在已找到 `N=0` 之后的局部窗口压缩里。它先保留原有 suffix post-polish，再由 QUBO 从停滞、回升、振荡窗口里选择少量 bridge 目标；只有找到严格更短且 validator 通过的替换才接受。

当前轻量 profile 参数为：

- `post_polish_mode=qubo_windows_v2`
- `post_polish_accept=better`
- `post_polish_window_select=1`
- `post_polish_window_max_size=8`
- `post_polish_window_repair_expanded=3000`

内部 30 复跑结果：30/30、509 步；窗口候选 1625 个，QUBO 选择 30 个，bridge 尝试 30 次，成功 0 次，窗口节省 0 步。总时间从同代码 `speed_v2` 复跑的 63.758 秒升到 71.844 秒。结论是：这个机制是安全的，但当前没有质量收益，不进入主线。

## 内部 30 算例

同机重跑，均要求 `final_n=0`。

| profile | 解出 | 总步数 | 总时间 sec | 平均时间 sec | QUBO vars | post saved | window saved |
|---|---:|---:|---:|---:|---:|---:|---:|
| `speed_v2` earlier probe | 30/30 | 509 | 58.113 | 1.937 | 22554 | 1 | 0 |
| `speed_v2` current rerun | 30/30 | 509 | 63.758 | 2.125 | 22555 | 1 | 0 |
| `post_polish_v1` | 30/30 | 509 | 59.239 | 1.975 | 22558 | 1 | 0 |
| `frontier_quality_v1` | 30/30 | 510 | 79.037 | 2.635 | 60624 | 1 | 0 |
| `stable+qubo_bucket` portfolio | 30/30 | 508 | 115.057 | 3.835 | - | - | - |
| `window_polish_v2` current rerun | 30/30 | 509 | 71.844 | 2.395 | 23819 | 1 | 0 |

内部 30 上，`post_polish_v1` 基本持平；`frontier_quality_v1` 多用约 2.7 倍 QUBO vars，却多 1 步且更慢。`stable+qubo_bucket` portfolio 通过择优少 1 步，但需要跑两套算法，总时间翻倍左右。
`window_polish_v2` 与 `speed_v2 current rerun` 是本次同代码对照：没有减少步数，主要增加了 bridge 验证时间。

## 外部 DLTS 7x5 cv2/cv3

范围：`7x5_cv2_test` 前 10 个 + `7x5_cv3_test` 前 10 个，`time_limit=10`。

| profile | 解出 | 成功算例总步数 | 总时间 sec | 成功算例平均 gap | QUBO vars |
|---|---:|---:|---:|---:|---:|
| `speed_v2` | 19/20 | 686 | 51.358 | 5.842 | 27721 |
| `post_polish_v1` | 19/20 | 685 | 62.713 | 5.789 | 27754 |
| `frontier_quality_v1` | 19/20 | 688 | 72.891 | 5.947 | 75409 |

外部 7x5 上，`post_polish_v1` 少 1 步、gap 略降，但时间更高，也没有救出失败的 `7x5_cv2_test_9`。`frontier_quality_v1` 成功率不变，步数和 gap 都变差，QUBO 变量数显著增加。

## 触发式 rescue 结果

范围：`7x5_cv1_test`、`7x5_cv2_test`、`7x5_cv3_test` 各前 10 个，`time_limit=10`。

| family | 解出 | 成功算例总步数 | 平均时间 sec | 成功算例平均 gap | rescue layers |
|---|---:|---:|---:|---:|---:|
| `7x5_cv1_test` | 7/10 | 268 | 4.749 | 7.857 | 108 |
| `7x5_cv2_test` | 10/10 | 380 | 4.622 | 6.800 | 78 |
| `7x5_cv3_test` | 10/10 | 350 | 3.952 | 4.900 | 68 |

相对之前记录的 `speed_v2` 外部结果，`rescue_qubo_v1` 的主要收益是解出率：`7x5_cv2_test` 前 10 个从 9/10 到 10/10，硬例 `7x5_cv2_test_9` 被救成 51 步 `N=0`。代价也明确：步数 gap 和平均时间升高，所以它是 hard-case rescue profile，不是质量主线。

## 结论

- `speed_v2` 仍是默认主线。
- `post_polish_v1` 可以保留为轻量质量试验 profile，但当前证据只支持“可能略降步数”，不支持替代主线。
- `frontier_quality_v1` 保留为负结果复现 profile，不推荐作为后续调参主线。
- `light_rescue_qubo_v1` / `scheduler_v1` 保留为开放 7x5 解出率增强路线；它们证明“触发式早 QUBO + 局部加宽”能救出普通 frontier QUBO gate 触达不到的 `N` 平台硬例，但不应写成步数质量更优。
- `qubo_pair_policy` 保留为 QUBO 贡献拆解工具；初步 smoke 未显示复杂 pairwise policy 的直接收益。
- `qubo_bucket` 保留为速度型挑战者和 no-regret portfolio 成员；内部 30 可通过择优少 1 步，但外部 7x5 hard case 不救。
- `no-regret portfolio` 是安全外壳，不是单算法加速；它适合报告“QUBO 算法簇最优解”，但 runtime 应按总组合时间报告。
- `protected_group_medium` 已确认能让 group QUBO 真正参与，但 A2_26 smoke 同步数且更慢，不作为当前主推。
- `window_polish_v2` 已确认安全但无收益：30 次窗口尝试没有接受替换，因此不作为当前主线。
- 简单扩大 frontier QUBO 变量、调用次数、补充宽度，并加入一步未来 `N` 偏置，并没有证明 QUBO 更强；反而显示当前瓶颈更可能在候选生成/搜索结构，而不是继续加大 QUBO 选择器。

## 复现命令

```powershell
python scripts\benchmark_compact_beam_qubo_n0.py --solver-profile speed_v2 --output docs\experiments\speed_v2_full30_rerun_probe.csv
python scripts\benchmark_compact_beam_qubo_n0.py --solver-profile post_polish_v1 --output docs\experiments\post_polish_quality_full30_probe.csv
python scripts\benchmark_compact_beam_qubo_n0.py --solver-profile frontier_quality_v1 --output docs\experiments\frontier_quality_full30_probe.csv
python scripts\benchmark_compact_beam_qubo_n0.py --solver-profile window_polish_v2 --output docs\experiments\window_polish_v2_full30_probe.csv
```

```powershell
python scripts\open_cpmp_benchmarks.py --families 7x5_cv2_test 7x5_cv3_test --limit-per-family 10 --variant qubo --solver-profile speed_v2 --time-limit 10 --output docs\experiments\open_cpmp_7x5_cv23_speed_v2_probe.csv
python scripts\open_cpmp_benchmarks.py --families 7x5_cv2_test 7x5_cv3_test --limit-per-family 10 --variant qubo --solver-profile post_polish_v1 --time-limit 10 --output docs\experiments\open_cpmp_7x5_cv23_post_polish_v1_probe.csv
python scripts\open_cpmp_benchmarks.py --families 7x5_cv2_test 7x5_cv3_test --limit-per-family 10 --variant qubo --solver-profile frontier_quality_v1 --time-limit 10 --output docs\experiments\open_cpmp_7x5_cv23_frontier_quality_v1_probe.csv
python scripts\open_cpmp_benchmarks.py --families 7x5_cv2_test --limit-per-family 10 --variant qubo --solver-profile rescue_qubo_v1 --time-limit 10 --output docs\experiments\open_cpmp_rescue_qubo_v1_7x5_cv2_10cases_probe.csv
python scripts\open_cpmp_benchmarks.py --families 7x5_cv1_test 7x5_cv3_test --limit-per-family 10 --variant qubo --solver-profile rescue_qubo_v1 --time-limit 10 --output docs\experiments\open_cpmp_rescue_qubo_v1_7x5_cv13_20cases_probe.csv
```

```powershell
foreach ($policy in "baseline","none","parent_only","stack_signature","hybrid") {
  python scripts\benchmark_compact_beam_qubo_n0.py --cases A2_26 --qubo-influence-profile light --qubo-pair-policy $policy --output docs\experiments\pair_policy_smoke_$policy.csv
}
```

```powershell
python scripts\benchmark_compact_beam_qubo_n0.py --cases A2_26 --selection-policy qubo_bucket --qubo-influence-profile light --output docs\experiments\qubo_bucket_a2_26_smoke.csv
python scripts\open_cpmp_benchmarks.py --families 7x5_cv2_test --limit-per-family 10 --variant qubo_bucket --time-limit 10 --output docs\experiments\open_cpmp_7x5_cv2_qubo_bucket_10cases.csv
python scripts\benchmark_qubo_no_regret_portfolio_n0.py --portfolio-profiles stable,qubo_bucket --output docs\experiments\no_regret_stable_bucket_full30.csv
```
