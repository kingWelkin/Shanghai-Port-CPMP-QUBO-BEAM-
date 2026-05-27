# 翻箱堆叠业务算法与 QUBO 结合点简报

本文简要说明当前项目中用于堆场翻箱/预整理问题的主算法路线，以及 QUBO 模块在其中承担的作用。当前目标不是求完整 CPMP 终局最优，而是在业务约束

`G = M(Now) + N(Future) <= floor(1.3 * N(0))`

下，尽量降低最终 `N(Future)`。

## 1. 问题口径

当前业务版本采用如下定义：

- `M(Now)`：当前已经执行的翻箱操作数。
- `N(Future)`：当前堆场状态下仍然存在的逆序箱数。
- `G`：业务熵值，定义为 `M(Now) + N(Future)`。
- 约束：搜索过程中任意时刻都必须满足 `G <= 1.3 * N(0)`。
- 目标：在上述约束内，尽可能降低最终 `N(Future)`。

因此，这一问题不是传统意义上“最少搬移完成全场严格有序”的 CPMP exact optimization，而是带业务预算约束的改善型轨迹搜索问题。

## 2. 当前主算法框架

当前效果最好的路线是：

`beam + enhanced_beam + tuned QUBO branch`

它不是把三个算法混成一个动作投票器，而是一个 portfolio：

1. 分别运行不同搜索分支。
2. 每个分支都只生成合法翻箱动作。
3. 每条轨迹都检查业务约束 `G <= 1.3N0`。
4. 最后按最终 `N(Future)` 优先选择最好结果。

三个分支的分工如下：

| 分支 | 主要作用 | 风险 |
|---|---|---|
| `beam` | 稳定、快速，按评分函数逐层扩展 | 容易保留相似状态，过早收敛 |
| `enhanced_beam` | 多规则候选生成，保留结构差异更大的动作 | 有时保留的多样性状态质量不够高 |
| tuned `QUBO branch` | 在候选集合中选择低评分且低相似度的一组状态 | 运行更慢，单独使用也不是每个算例都最好 |

## 3. QUBO 的结合点

QUBO 当前没有直接建完整翻箱序列模型，也没有直接输出最终解。

当前 QUBO 的角色是：

> 对 beam search 每一层产生的合法候选状态做组合选择。

具体流程是：

1. 经典状态引擎先从当前 beam 中生成合法动作。
2. 每个动作经过模拟器 `apply_move` 检查：
   - 只能搬源栈顶箱；
   - 目标栈不能满；
   - 源栈和目标栈不同；
   - 中间状态必须满足业务熵约束。
3. 得到一批合法候选状态后，构建 QUBO。
4. QUBO 选择其中一组进入下一层 beam。
5. QUBO sample 若不满足选择数量要求，会 repair。
6. 最终轨迹仍由经典模拟器回放验证。

因此，QUBO 不承担可行性保证；可行性完全由经典状态引擎和 validator 保证。

## 4. QUBO 形式

对一层候选状态 `c_i`，建立二进制变量：

`u_i = 1` 表示选择候选状态 `c_i` 进入下一层 beam。

QUBO 目标包括三部分：

1. 候选状态评分  
   评分越低，说明当前 `N(Future)`、`G`、倒置数等指标越好。

2. 候选之间的相似度惩罚  
   如果两个状态中箱子所在栈的分配非常相似，同时选择它们的收益较低，因此加入 pairwise penalty。

3. beam 宽度约束  
   使用 penalty 约束选中的候选数量接近 beam width。

简化形式为：

`min sum_i score_i u_i + lambda * sum_{i<j} similarity(i,j) u_i u_j + A * (sum_i u_i - B)^2`

其中：

- `B` 是 beam width。
- `A` 是 one-hot / exactly-K penalty 强度。
- `lambda` 控制多样性。

这个 QUBO 的意义不是“直接判断哪一步最优”，而是在一层候选中选择一组互补状态，缓解普通 beam search 的候选坍缩问题。

## 5. 为什么 QUBO 有用

普通 beam search 往往只保留评分靠前的一批状态，但这些状态可能结构非常相似，导致后续搜索空间变窄。

QUBO branch 的价值在于：

- 既考虑候选状态质量；
- 又考虑候选之间的结构差异；
- 因而能保留一些 plain beam 可能提前删掉的潜在路线。

这对当前业务目标尤其重要，因为在 `G <= 1.3N0` 的预算内，有些路线短期看起来不占优，但后续可能显著降低 `N(Future)`。

## 6. 当前实验结果

在 15 个真实 `01+26` 配对业务算例上，当前结果如下：

| 算法路线 | 平均最终 `N(Future)` | 平均搬移数 | 平均运行时间 |
|---|---:|---:|---:|
| `chen_greedy_algorithm_1p3` | 17.13 | 20.20 | 0.115s |
| `beam` | 10.67 | 24.13 | 2.460s |
| `qubo_beam` 原始版 | 10.60 | 24.27 | 2.932s |
| `beam_focused_plain` | 9.47 | 25.47 | 11.286s |
| `qbo_beam_standalone_portfolio` | 9.00 | 26.87 | 26.027s |
| `beam_focused_with_qubo_portfolio` | 8.53 | 26.80 | 37.313s |

可以看到：

- QUBO 单独增强有效：`qubo_beam` 从 `10.60` 改进到 QUBO-only portfolio 的 `9.00`。
- QUBO 与 focused beam 组合后继续有效：无 QUBO 的 `beam_focused_plain` 为 `9.47`，加入 tuned QUBO branch 后达到 `8.53`。
- QUBO 的主要贡献是质量提升，不是速度提升。

## 7. 当前判断

当前推荐算法为：

`beam_focused_with_qubo_portfolio`

原因是它综合了三类互补搜索能力：

- plain beam 提供稳定基线；
- enhanced beam 增加规则多样性；
- tuned QUBO branch 提供结构多样性的候选选择。

QUBO 当前最适合作为候选选择和 portfolio 分支，不适合作为单独直接求解器。

## 8. 后续方向

后续可以继续推进三件事：

1. 提高 QUBO branch 的效率，减少不必要的 QUBO 调用。
2. 引入外部 QUBO / Ising API，但必须保留 decode、repair、validate。
3. 分析每个算例中 QUBO 胜出的具体结构，形成更可解释的业务规则。

