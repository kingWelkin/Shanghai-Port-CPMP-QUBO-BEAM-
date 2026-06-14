# 当前 QUBO Beam 主算法完整描述

本文说明当前主推的 `N=0` 单贝位算法。实现入口是：

```text
scripts/benchmark_compact_beam_qubo_n0.py
```

推荐使用 profile：

```text
speed_v2
```

当前微优化版结果：

| 指标 | 数值 |
|---|---:|
| 单贝位算例数 | 30 |
| `N=0` 解出 | 30/30 |
| 总步数 | 509 |
| 平均时间 | 2.237 秒 |
| 26 号贝位解出 | 15/15 |
| 26 号贝位总步数 | 414 |
| 26 号贝位平均时间 | 4.031 秒 |

## 1. 算法定位

当前算法不是 exact solver，也不证明最优步数。它是一个以 Beam search 为主体、QUBO 做少量候选补充和后处理选择的启发式搜索算法。

目标是让每个单贝位达到：

```text
final_n = 0
```

这里的 `N` 表示当前仍处于逆序影响范围内的箱子数量。

## 2. 合法动作和约束

每个状态是一个单贝位堆场，stack 按 bottom -> top 表示。

合法动作只有一种：

1. 从某个非空栈取走栈顶箱；
2. 放到另一个未满栈的栈顶；
3. 不允许移动非栈顶箱；
4. 不允许把箱子移出堆场；
5. 不允许超过最大栈高。

搜索过程还使用业务预算约束：

```text
当前搬移步数 + 当前 N <= floor(1.3 * 初始 N)
```

所有输出动作都要经过 validator 重新模拟。QUBO 不直接产生动作，也不直接声明可行解。

## 3. 状态表示

主搜索使用 compact byte-state：

- 每个栈编码成固定长度 bytes；
- 所有栈拼接成状态码；
- 等价栈排列做 canonical 排序；
- 路径找到后再 lift 回真实物理栈动作。

当前微优化包括：

- `_cached_chunk_meta`：缓存单栈高度和逆序贡献；
- `_cached_pop_chunk` / `_cached_push_chunk`：缓存单栈出栈/入栈后的 bytes；
- 默认 `n_then_depth` 排序走轻量 key。

这些优化不改变搜索策略，只降低 Python 层开销。

## 4. 主搜索流程

主搜索是分层 Beam：

1. 从初始状态计算初始 `N` 和业务上限。
2. 对当前 frontier 中的每个状态枚举所有合法栈顶移动。
3. 增量计算后继状态的 `N`。
4. 剪掉违反业务预算的候选。
5. 剪掉已用更短深度访问过的 dominated 状态。
6. 如果出现 `N=0` 候选，按排序规则选出目标并停止主搜索。
7. 否则选择下一层 frontier。

默认排序是：

```text
(N, depth, parent_code, state_code)
```

因此主搜索强优先降低 `N`，再比较步数和稳定 tie-break。

## 5. Frontier 选择

`speed_v2` 使用 enhanced selection：

```text
beam_width = 1200
core_width = 1080
parent_core_cap = 4
selection_policy = enhanced
```

### 5.1 deterministic core

每层先按排序选出 deterministic core。`parent_core_cap=4` 会限制同一个父状态最多贡献 4 个子状态，避免少数父状态的相似后继挤满 frontier。

这一部分不依赖 QUBO。它也是 enhanced no-QUBO 能达到 30/30 的主要原因。

### 5.2 frontier QUBO supplement

在 core 之外，QUBO 对候选状态做小比例补充：

```text
qubo_mode = focused_multi
max_qubo_calls = 32
max_qubo_vars = 64
qubo_supplement_width = 24
qubo_focus_n = 8
qubo_cohorts_per_layer = 2
```

QUBO 变量表示候选状态是否进入补充集合：

```text
x_i = 1 表示选择第 i 个候选状态
```

目标函数包含：

- 候选质量项：偏好低 `N`、浅 depth、高 rank 候选；
- 二次惩罚项：惩罚过相似、同父或相近候选同时入选；
- exactly-k penalty：控制选中数量。

采样后仍会 repair 到合法数量。repair 后的结果只是 frontier 候选，后续动作可行性仍由 classical state engine 保证。

## 6. Post-polish QUBO

主搜索找到 `N=0` 后，post-polish 尝试缩短路径。

它分为：

- suffix polish：从低 `N` 尾部某个状态重新搜索更短到达 `N=0` 的后缀；
- window polish：尝试用更短 bridge 替换已有路径局部窗口。

QUBO 在这里选择值得尝试的 suffix/window。它不直接改写路径；每个替换都必须重新通过 lift 和 validator。

当前证据表明：

- Post-polish 主要贡献是步数压缩。
- `micro_optimized_speed_v2_full30.csv` 中总共节省 1 步。
- 明确案例是 `A2_26` 从 29 步压到 28 步。
- Post-polish 不负责把 `N>0` 失败路径变成 `N=0`。
- `window_polish_v2` 已实现为只接受严格更短 bridge 的安全后处理；内部 30 复跑中窗口尝试 30 次、成功 0 次、总步数仍为 509，因此保留为实验 profile，不进入主线。

## 7. QUBO 的实际贡献边界

当前最稳妥的解释是：

- 相对 strict classic no-QUBO，QUBO Beam 搜索框架显著提升归零稳定性：30/30 对 25/30。
- 相对 enhanced no-QUBO，frontier QUBO 对 30/30 不是唯一必要条件，因为 enhanced no-QUBO 也能 30/30。
- Post-polish QUBO 的可见贡献是总步数 510 到 509 的 1 步压缩。
- 最终路径上 QUBO direct hit 很少，说明它不是逐步主导路径，而是稀疏影响 frontier 结构。
- 简单扩大 QUBO 影响并不自动带来收益：`frontier_quality_v1` 更慢且不优，`window_polish_v2` 安全但没有接受替换。

因此推荐写成：

> QUBO 是稀疏前沿纠偏和后处理压缩模块，不是可行性引擎。可行性由 classical state engine、Beam 搜索和 validator 保证。

## 8. Dynamic QUBO 变体

dynamic QUBO 使用 `selection_policy=dynamic_qubo`，不固定每层 frontier 为 `beam_width=1200`，而是在最小、核心和最大 frontier 之间调节规模。

当前保留为质量增强变体：

| 变体 | 解出 | 总步数 | 平均时间 |
|---|---:|---:|---:|
| dynamic QUBO quality | 30/30 | 507 | 4.902 秒 |
| dynamic QUBO `balanced2200` | 30/30 | 508 | 4.137 秒 |

dynamic QUBO 可以减少总步数，但慢于当前非 dynamic `speed_v2`，因此不作为速度主线。

## 9. Trigger Rescue QUBO 变体

`rescue_qubo_v1` 使用 `selection_policy=rescue_qubo`，不是默认主线。它只在风险层临时增强 QUBO：

```text
best_n plateau window = 2
cap slack <= 26
min candidates = 5000
max rescue layers = 16
local beam/core = 1350/1215
after 2 rescue layers escalate to 1600/1440
temporary qubo_focus_n = -1
```

它解决的是普通 `speed_v2` 中 QUBO gate 触发太晚的问题。例如 `7x5_cv2_test_9` 的失败路径长期停在 `N=16`，高于 `speed_v2` 的 `qubo_focus_n=8`，因此 frontier QUBO 原本没有机会介入。

当前开放 7x5 前 10 例结果：

| family | 解出 | 成功算例总步数 | 平均时间 |
|---|---:|---:|---:|
| `7x5_cv1_test` | 7/10 | 268 | 4.749 秒 |
| `7x5_cv2_test` | 10/10 | 380 | 4.622 秒 |
| `7x5_cv3_test` | 10/10 | 350 | 3.952 秒 |

这个 profile 的定位是 hard-case rescue：它提高部分开放 7x5 的解出率，但会增加时间和参考 gap，不应替代 `speed_v2` 作为上港 30 算例主线。

## 10. Adaptive QUBO 变体

`selection_policy=adaptive_qubo` 是开放 7x5 cv1 的速度/解出率 profile。它在高 `N` 阶段收窄 frontier，让搜索更快下潜；在低 `N` 末段恢复到 1200，保留 QUBO 和后处理质量。

当前 7x5 cv1 40 例、`time_limit=3` 的结果：

| 变体 | 解出 | 成功算例总步数 | 平均时间 | 成功算例平均 gap | 定位 |
|---|---:|---:|---:|---:|---|
| `qubo_adaptive` | 22/40 | 836 | 2.014 秒 | 6.955 | 当前 cv1 解出率/速度基线 |
| `qubo_adaptive_polish` | 22/40 | 834 | 2.029 秒 | 6.864 | 质量小增强，当前推荐 cv1 变体 |
| `qubo_adaptive_rescue` early | 20/40 | 746 | 2.185 秒 | 6.100 | 早救援误伤已解 case，不推荐 |
| `qubo_adaptive_rescue` late | 22/40 | 835 | 2.021 秒 | 6.909 | 轻微质量收益，没有新增解出 |

因此，`adaptive_polish_v1` 是目前 `qubo_adaptive` 的保守增强：不提高解出率，但在同样 22/40 上少 2 步。`adaptive_rescue_v1` 保留为调参入口；当前证据说明“把 rescue 接到 adaptive”必须很晚、很轻，否则会打坏已解路径。

新增 `adaptive_controller_v1` 是结构感知 adaptive 试验，不再只按 `best_n` 分段，而是同时看候选压力、plateau、depth slack 和实例栈数。本轮本地复测中，它在 7x5 cv1 test+validation 从 old `qubo_adaptive` 的 19/40 提到 21/40，7x5 cv2+cv3 test 保持 18/20 且平均时间从 2.241 秒降到 1.939 秒；5x5 cv1 保持 19/40。内部 30 则只与 old `adaptive_qubo` 持平 27/30，并从 1.768 秒升到 1.857 秒。因此它是开放 7x5 候选 profile，不是内部主线替代品。

## 11. 实例感知 scheduler 变体

`scheduler_v1` 使用 `selection_policy=scheduler_qubo`。它不是一个新的搜索引擎，而是对已有 profile 的调度：

- 默认沿用 `speed_v2` 的 enhanced selection；
- 当 `best_n` 平台、候选数压力和 cap slack 同时满足时，触发 `light_rescue_qubo_v1`；
- adaptive 起步不读取算例名，而是检查重复优先级比例、`best_n / initial_n`、候选压力、父节点集中度和 plateau/slack；
- 重复优先级实例默认不启用 adaptive 起步，避免在 priority-group 硬例族上误伤；唯一优先级且高 `N` 的 7-stack 层仍可使用 adaptive frontier。

当前 `7x5` 前 10 个、`time_limit=5` 的 signal-aware 小矩阵显示：

| family | 变体 | 解出 | 成功算例总步数 | 平均时间 | 说明 |
|---|---|---:|---:|---:|---|
| `7x5_cv1_test` | `scheduler_v1` signal-aware | 6/10 | 222 | 2.958 秒 | unique-priority hard case 仍会失败 |
| `7x5_cv2_test` | `scheduler_v1` signal-aware | 10/10 | 381 | 2.925 秒 | 保住 hard priority-group case |
| `7x5_cv3_test` | `scheduler_v1` signal-aware | 10/10 | 354 | 2.486 秒 | 无明显副作用 |

其中 `7x5_cv2_test` 前 10 的同口径对照为：

| 变体 | 解出 | 成功算例总步数 | 平均时间 | 说明 |
|---|---:|---:|---:|---|
| `speed_v2` | 9/10 | 330 | 3.642 秒 | `7x5_cv2_test_9` 未归零 |
| `light_rescue_qubo_v1` | 10/10 | 379 | 3.853 秒 | 救出 hard case，但步数 gap 增大 |
| `scheduler_v1` signal-aware | 10/10 | 381 | 2.925 秒 | 当前最值得继续扩大矩阵的开放算例候选 |
| `scheduler_topk_v1` signal-aware | 10/10 | 381 | 3.280 秒 | `rank_truncated=0`，top-k 未触发真实裁剪 |

因此，`scheduler_v1` 的定位是开放算例解出率路线，不替代内部 30 默认主线 `speed_v2`。它已经从按 `cv1/cv2/cv3` 名称调参改为按实例结构和层级信号调度，但 `7x5_cv1` 失败例说明还不能写成通用 profile。

## 12. Bucket QUBO 与 no-regret portfolio

`selection_policy=qubo_bucket` 是候选保留策略，不是默认主线。它把已排序候选按 `N/depth`、`depth+N` 或 `N` band 分桶；每个桶内先保留 deterministic core，再让 QUBO 选择少量补充候选。

它的作用边界是：

- 不跨桶推翻主启发式排序；
- 只在近似等价候选内部增加 QUBO 选择权；
- 可减少 QUBO 调用次数和部分 case runtime；
- 外部 7x5 hard case 上不能替代 `rescue_qubo_v1`。

新增 no-regret portfolio 脚本：

```text
scripts/benchmark_qubo_no_regret_portfolio_n0.py
```

它先跑 `stable`，再跑指定 QUBO 挑战者，只选择验证通过且更短的路径。内部 30 的 `stable+qubo_bucket` 结果为 30/30、508 步，比 `speed_v2` baseline 少 1 步；但总组合时间为 115.057 秒，必须按 portfolio 总时间报告。

## 13. 复现命令

主线：

```powershell
python scripts\benchmark_compact_beam_qubo_n0.py --profile speed_v2 --output docs\experiments\micro_optimized_speed_v2_full30.csv
```

enhanced no-QUBO：

```powershell
python scripts\benchmark_compact_beam_qubo_n0.py --profile speed_v2 --no-use-qubo --no-post-polish-qubo --selection-policy enhanced --beam-width 1200 --core-width 1080 --max-qubo-calls 0 --output docs\experiments\micro_optimized_strict_noqubo_enhanced_w1200_full30.csv
```

strict classic no-QUBO 时间匹配：

```powershell
python scripts\benchmark_compact_qubo_ablation_time_aligned.py --solver-profile speed_v2 --sweep-all-no-qubo --output docs\experiments\micro_optimized_strict_ablation_full30.csv --no-qubo-widths 1300 1400 1600 2000 2400 3200 4800 6400
```

外部 7x5 hard-case rescue：

```powershell
python scripts\open_cpmp_benchmarks.py --families 7x5_cv2_test --limit-per-family 10 --variant qubo --solver-profile rescue_qubo_v1 --time-limit 10 --output docs\experiments\open_cpmp_rescue_qubo_v1_7x5_cv2_10cases_probe.csv
```

scheduler / light rescue：

```powershell
python scripts\open_cpmp_benchmarks.py --families 7x5_cv2_test --limit-per-family 10 --variant qubo_light_rescue --time-limit 5 --output docs\experiments\scratch_7x5_cv2_10_light_rescue.csv
python scripts\open_cpmp_benchmarks.py --families 7x5_cv2_test --limit-per-family 10 --variant qubo_scheduler --time-limit 5 --output docs\experiments\scratch_signal_7x5_cv2_scheduler.csv
```

adaptive QUBO：

```powershell
python scripts\open_cpmp_benchmarks.py --families 7x5_cv1_test 7x5_cv1_validation --limit-per-family 20 --variant qubo_adaptive --time-limit 3 --alpha 10 --output docs\experiments\open_cpmp_dlts_7x5_cv1_40cases_adaptive.csv
python scripts\open_cpmp_benchmarks.py --families 7x5_cv1_test 7x5_cv1_validation --limit-per-family 20 --variant qubo_adaptive_polish --time-limit 3 --alpha 10 --output docs\experiments\open_cpmp_dlts_7x5_cv1_40cases_adaptive_polish.csv
```

bucket QUBO 与 no-regret portfolio：

```powershell
python scripts\benchmark_compact_beam_qubo_n0.py --cases A2_26 --selection-policy qubo_bucket --qubo-influence-profile light --output docs\experiments\qubo_bucket_a2_26_smoke.csv
python scripts\benchmark_qubo_no_regret_portfolio_n0.py --portfolio-profiles stable,qubo_bucket --output docs\experiments\no_regret_stable_bucket_full30.csv
```

测试：

```powershell
python -m pytest -q
```
