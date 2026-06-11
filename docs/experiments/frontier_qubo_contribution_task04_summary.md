# Task 04：QUBO 贡献拆解摘要

本轮目标是把“QUBO 到底贡献了什么”拆成搜索阶段和后处理阶段两部分，避免把 `N=0`、少步数、时间优势混成一个结论。

## 1. 搜索阶段 Frontier QUBO

当前主线 `speed_v2` 的搜索结构是：

- `beam_width=1200`
- `core_width=1080`
- `parent_core_cap=4`
- `qubo_mode=focused_multi`
- `qubo_supplement_width=24`
- `qubo_cohorts_per_layer=2`

也就是说，QUBO 不是主搜索引擎。每层大部分 frontier 来自 deterministic enhanced Beam，QUBO 只补充少量候选状态。

在 `micro_optimized_speed_v2_full30.csv` 中：

| 指标 | 数值 |
|---|---:|
| 解出 | 30/30 |
| 总步数 | 509 |
| frontier QUBO 总调用 | 366 |
| 直接落在最终路径上的 QUBO 选择 | 2 |
| 有 QUBO 祖先关系的最终路径状态 | 21 |

直接路径贡献主要集中在：

| 算例 | moves | QUBO direct on path | QUBO ancestor on path |
|---|---:|---:|---:|
| `C1_26` | 21 | 1 | 10 |
| `C2_26` | 31 | 1 | 11 |

因此，frontier QUBO 的合理表述不是“逐步主导最终路径”，而是：

> QUBO 以小比例补充候选，稀疏地改变搜索前沿结构；在部分难例上，它的候选或其后代进入最终路径。

## 2. Post-Polish QUBO

Post-polish 在主搜索已经找到 `N=0` 后才运行。它不能把失败路径变成 `N=0`，只能尝试缩短已找到的合法 `N=0` 路径。

当前证据：

- `micro_optimized_speed_v2_full30.csv`：post-polish 总共节省 1 步。
- 明确动作级案例：`A2_26` 从 29 步缩到 28 步。
- `micro_optimized_strict_noqubo_enhanced_w1200_full30.csv`：关闭 frontier QUBO 和 post QUBO 后，enhanced no-QUBO 仍为 30/30，但总步数为 510。

因此 Post 的贡献应写成：

> Post-polish QUBO 的主要贡献是局部步数压缩，目前稳定证据是总步数 510 到 509 的 1 步改进；它不是 `N=0` 归零的来源。

## 3. 与不同 baseline 的解释边界

相对 strict classic no-QUBO：

- QUBO `speed_v2`：30/30。
- classic no-QUBO time-matched：25/30。
- 26 号贝位：QUBO 15/15，classic no-QUBO 10/15。

这个对比说明：QUBO Beam 搜索框架相对传统 Beam 有明显归零稳定性优势。

相对 enhanced no-QUBO：

- enhanced no-QUBO 也能 30/30。
- 但总步数为 510，QUBO 主线为 509。

这个对比说明：当前代码中 deterministic enhanced 机制本身很强，frontier QUBO 对 30/30 不是唯一必要条件；QUBO 模块更适合被描述为稀疏纠偏和局部压缩模块。

## 4. 推荐表述

建议主文档采用以下口径：

> 当前 QUBO Beam 不是把可行性完全交给 QUBO。可行性由 classical state engine、Beam 搜索、业务约束和 validator 保证。QUBO 在搜索阶段以小比例补充候选、改善 frontier 结构；在后处理阶段选择局部 suffix/window 做路径压缩。严格时间匹配下，相对 classic Beam，QUBO Beam 从 25/30 提升到 30/30；相对 enhanced no-QUBO，QUBO 的可见贡献主要是 1 步路径压缩和少量路径祖先影响。
