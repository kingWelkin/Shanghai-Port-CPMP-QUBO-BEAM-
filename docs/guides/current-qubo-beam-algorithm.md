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

## 7. QUBO 的实际贡献边界

当前最稳妥的解释是：

- 相对 strict classic no-QUBO，QUBO Beam 搜索框架显著提升归零稳定性：30/30 对 25/30。
- 相对 enhanced no-QUBO，frontier QUBO 对 30/30 不是唯一必要条件，因为 enhanced no-QUBO 也能 30/30。
- Post-polish QUBO 的可见贡献是总步数 510 到 509 的 1 步压缩。
- 最终路径上 QUBO direct hit 很少，说明它不是逐步主导路径，而是稀疏影响 frontier 结构。

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

## 9. 复现命令

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

测试：

```powershell
python -m pytest -q
```
