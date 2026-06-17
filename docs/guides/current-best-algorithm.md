# 当前最佳算法说明：QUBO Scheduler State V5 Wang Rescue Polish

日期：2026-06-15

本文回答一个容易混淆的问题：当前项目里“最好的算法”到底是哪一个。

结论是：

- 若评价目标是 **open DLTS 5x5/7x5 test-120 的单一完整算法质量**，当前最好的是 `qubo_scheduler_state_v5_wang_rescue_polish`。
- 若评价目标是 **上港内部 30 个单贝位的默认运行主线**，仍应优先使用 `speed_v2` / `scheduler_state_v5` 这一类轻量主线。
- 若评价目标是 **open DLTS 全覆盖可行性**，`wang_bs_b` 能做到 120/120，但路径明显更长，因此更适合作为 fallback 和教师，不是当前质量主算法。

下面的说明以当前最好的单一质量算法 `qubo_scheduler_state_v5_wang_rescue_polish` 为主。

## 1. 一句话定义

`qubo_scheduler_state_v5_wang_rescue_polish` 是一个以 compact byte-state Beam search 为主体的启发式 CPMP / pre-marshalling 算法。它在普通 QUBO scheduler 的基础上加入 Wang 启发式 rescue，并在找到可行 `N=0` 路径后用 QUBO 选择 suffix/window 做局部路径压缩。

它不是纯 QUBO 求解器，也不是 exact solver。QUBO 在其中承担候选选择、rescue 补充和后处理选择职责；动作可行性、路径重建和最终验收仍由 classical state engine 和 validator 负责。

## 2. 当前证据

在 open DLTS 5x5/7x5 test-120 口径下，当前整理后的 single-profile 排名如下：

| 算法 | 解出 | full-quality penalty | 成功解总步数 | 成功解 gap sum | 平均时间 |
|---|---:|---:|---:|---:|---:|
| `qubo_scheduler_state_v5_wang_rescue_polish` | 118/120 | 3873 | 3692 | 350 | 3.901 秒 |
| `qubo_scheduler_state_v5_wang_rescue_suffix_polish` | 118/120 | 3880 | 3699 | 357 | 3.883 秒 |
| `qubo_scheduler_state_v5_wang_rescue` | 118/120 | 3887 | 3706 | 364 | 3.551 秒 |
| `qubo_scheduler_state_v5_wang_features` | 105/120 | 5162 | 3368 | 455 | 2.749 秒 |
| `qubo_scheduler_state_v5` | 89/120 | 6334 | 2856 | 395 | 1.941 秒 |
| `wang_bs_b` | 120/120 | 4569 | 4569 | 1174 | 1.663 秒 |

主要解释：

- `wang_bs_b` 解出率最高，120/120，但路径质量差，gap sum 很高。
- `qubo_scheduler_state_v5_wang_rescue_polish` 是当前最好的单一质量算法：118/120，full-quality penalty 最低。
- 在 `wang_rescue_polish` 和 `wang_bs_b` 都解出的 118 个 case 上，`wang_rescue_polish` 用 3692 步，Wang BS-B 用 4496 步，少 804 步；其中 QUBO-rescue-polish 在 109 个 case 上更短，Wang 只在 5 个 case 上更短，4 个持平。
- 当前剩余两个未解 case 是 `5x5_cv1_test_4` 和 `5x5_cv2_test_10`。

参考文件：

- `docs/experiments/wang_qubo_direction_summary.md`
- `docs/experiments/wang_qubo_full_quality_trial_summary.md`

## 3. 为什么它是“当前最好”

这里的“最好”不是指所有指标都第一，而是指在当前最重要的 full-case quality 口径下，它是最强的单一算法。

### 3.1 它不是 portfolio 上界

per-case portfolio 可以先跑多个算法，再挑每个 case 的最好结果。这种方式可以得到更好的汇总数字，但它不是一个单一算法，也不能代表一次真实部署运行的成本。

`qubo_scheduler_state_v5_wang_rescue_polish` 是单 profile，可以作为完整算法运行和复现。

### 3.2 它比 Wang BS-B 更适合做质量主线

Wang BS-B 的优势是可行性和速度：它在 test-120 上 120/120。但它为了可行性接受了较长路径，总步数 4569，gap sum 1174。

`wang_rescue_polish` 的覆盖率低 2 个 case，但成功路径显著更短。对于当前项目的算法研究目标，它更像质量主线；Wang BS-B 更像兜底基线和失败样例教师。

### 3.3 polish 带来了小但真实的质量收益

从 rescue 到 rescue-polish：

| 算法 | 解出 | full-quality penalty | 成功解总步数 | gap sum |
|---|---:|---:|---:|---:|
| `scheduler_state_v5_wang_rescue` | 118/120 | 3887 | 3706 | 364 |
| `scheduler_state_v5_wang_rescue_polish` | 118/120 | 3873 | 3692 | 350 |

polish 没有提高解出数，但把 full-quality penalty 降了 14，把成功解总步数降了 14，把 gap sum 降了 14。这说明后处理不是装饰项，而是在 rescue 找到可行路之后回收了一部分路径质量。

## 4. 算法结构

当前最佳算法可以分成五层：

1. compact state engine
2. scheduler-state Beam 主搜索
3. QUBO frontier supplement
4. Wang rescue / pair-QUBO hard-case supplement
5. suffix/window post-polish QUBO

### 4.1 compact state engine

主搜索不直接维护完整对象图，而是使用 compact byte-state：

- 每个栈编码成固定长度 bytes；
- 所有栈拼接成一个状态码；
- 等价栈排列做 canonical 排序；
- 搜索时保存 parent code，找到路径后再 lift 回真实栈动作。

这样做的目的有两个：

- 降低 Python 搜索中的状态比较和哈希成本；
- 让 dominated-state 剪枝、parent-chain 重建和 telemetry 更稳定。

动作仍然只有合法栈顶移动：

1. 从非空栈取栈顶箱；
2. 放入另一个未满栈；
3. 不移动非栈顶；
4. 不移出堆场；
5. 不超过栈高。

### 4.2 scheduler-state Beam 主搜索

基础主线来自 `scheduler_state_v5`。它不是简单固定宽度 Beam，而是根据搜索层状态在几种模式之间切换：

- `contract`：候选压力大、搜索需要收缩时，降低有效 frontier；
- `protect`：保护低 `N` 和关键 parent，避免好路径被噪声挤掉；
- `rescue`：遇到 plateau 或高风险层时，临时扩大/增强候选选择。

`scheduler_state_v5` 相比早期版本的重要点是：在初始 plateau 阶段要求更高的 unique min-N 信号，避免过早 rescue，也避免被重复候选误导。

### 4.3 QUBO frontier supplement

QUBO 不是主搜索引擎。普通搜索层里，算法先保留 deterministic core，再用 QUBO 从剩余候选中补充少量状态。

QUBO 变量含义是：

```text
x_i = 1 表示第 i 个候选状态进入补充集合
```

目标函数主要包含：

- 候选质量项：低 `N`、浅 depth、较好 rank 更优；
- 多样性二次项：惩罚过相似、同父或结构近似的候选同时入选；
- exactly-k penalty：控制最终选择数量。

采样结果如果数量不合法，还会 repair 到目标数量。repair 之后的结果也只是下一层 frontier 候选，不直接生成最终动作。

### 4.4 Wang rescue / pair-QUBO supplement

`scheduler_state_v5_wang_rescue_polish` 继承 `scheduler_state_v5_wang_rescue`。这一层加入 Wang-inspired rescue：

- `wang_pair_qubo_enabled = True`
- `wang_pair_qubo_supplement_width = 12`
- `wang_pair_qubo_focus_n = -1`
- `wang_pair_qubo_max_vars = 48`
- `wang_pair_qubo_select = 12`
- `wang_pair_qubo_same_parent_penalty = 0.35`

它的作用不是把 Wang 路径直接拼进 compact 路径，而是在风险层给 Beam 提供更像 Wang fixed-prefix / fix-pair 判断的候选补充。

这点很重要：算法仍然保留 compact parent-chain，不接受外部路径硬拼。任何补充状态都必须能由当前 compact parent 合法转移得到。

### 4.5 rescue QUBO 增强

在 rescue 层，算法临时增强 QUBO 选择能力：

- `rescue_qubo_supplement_width = 56`
- `rescue_max_qubo_vars = 112`
- `rescue_next_n_weight = 0.75`

这解决的是普通 QUBO gate 触发太晚的问题。部分 hard case 会长时间停在较高 `N` 平台，若只在低 `N` 阶段触发 QUBO，算法根本没有机会介入早期逃逸。

rescue 的定位是：在 plateau / high-risk 层临时增加探索质量，而不是全局无条件放大 QUBO。

### 4.6 post-polish QUBO

主搜索找到 `N=0` 后，算法进入 post-polish：

- `post_polish_qubo = True`
- `post_polish_mode = qubo_windows_v2`
- `post_polish_accept = better`
- `post_polish_passes = 3`
- suffix 候选最多 72，QUBO 选 7；
- window QUBO 每轮选 2 个窗口，最多 72 个变量；
- bridge repair 展开上限为 6000。

post-polish 只接受严格更短且验证通过的替换。它不能把 `N>0` 失败路径变成成功路径，只负责压缩已经可行的 `N=0` 路径。

## 5. 运行流程

整体流程如下：

1. 读取 DLTS 或内部单贝位实例。
2. 计算初始 `N`、容量、业务/质量预算和目标语义。
3. 用 compact bytes 编码初始状态。
4. 分层展开所有合法栈顶移动。
5. 剪掉违反容量、业务预算或 dominated-depth 的候选。
6. 根据 scheduler-state 信号决定本层是 contract、protect 还是 rescue。
7. 保留 deterministic core，并用 QUBO / Wang pair-QUBO 补充少量 frontier。
8. 若发现 `N=0` 路径，则停止主搜索。
9. 对可行路径运行 suffix/window post-polish。
10. lift 回真实动作序列。
11. 用 validator 重新模拟动作，记录 `business_ok`、`goal_validation_ok`、moves、gap、QUBO 计数和 polish 计数。

## 6. 与其他算法的关系

### 6.1 与 `speed_v2`

`speed_v2` 是内部 30 个单贝位的默认主线：

| 算法 | 内部 30 解出 | 总步数 | 平均时间 | 定位 |
|---|---:|---:|---:|---|
| `speed_v2` | 30/30 | 509 | 2.237 秒 | 内部默认主线 |
| `scheduler_state_v5_wang_rescue` | 30/30 | 508 | 约 198 秒总时间 | 质量 rescue，太慢 |

内部 30 上，Wang rescue 只带来 1 步级别收益，但运行成本明显更高。因此内部部署或紧时间预算下，不应无条件替换 `speed_v2`。

### 6.2 与 `scheduler_state_v5`

`scheduler_state_v5` 是 open DLTS 上的轻量 scheduler 基线。它快，但 hard-case coverage 不足：

```text
scheduler_state_v5: 89/120
wang_rescue_polish: 118/120
```

因此 `scheduler_state_v5` 适合作为基础结构，而不是当前 open120 的最终质量主线。

### 6.3 与 `wang_bs_b`

`wang_bs_b` 是强可行性 baseline：

```text
wang_bs_b: 120/120, 4569 moves, gap sum 1174
wang_rescue_polish: 118/120, 3692 moves, gap sum 350
```

更合理的策略是：

- 先运行 QUBO/scheduler 质量主线；
- 若失败，再用 Wang BS-B 兜底；
- 若两者都成功，优先保留更短的 QUBO/scheduler 路径。

### 6.4 与实验变体

这些方向已经试过，但暂不提升为主线：

| 方向 | 观察 | 结论 |
|---|---|---|
| `wang_rescue_suffix_polish` | 118/120，penalty 3880 | 比 full polish 少一点时间，但质量低 7 penalty |
| selective-window polish | clean run 117/120 | 能减少窗口尝试，但新增失败，不替代主线 |
| dynamic survivor cap | paired run更快但丢 coverage | 保留为实验开关 |
| risk group-QUBO polish | 107/120 | 7x5 不稳，不推广 |
| pair-QUBO only | 94/120 | 有覆盖提升，但远不够 |
| full Wang hybrid | 93/120 | 更重且更差 |
| Wang2017 task polish | 当前 scratch open120 很弱 | 不推广 |

## 7. 当前局限

当前最佳算法仍有明确边界：

1. 不是 exact solver，不证明最优步数。
2. open DLTS test-120 仍有 2 个未解 case。
3. 相比 Wang BS-B，覆盖率不是 100%。
4. 相比 `speed_v2`，运行成本更高，不适合作为内部 30 的默认快速主线。
5. QUBO 仍是选择器，不是完整 CPMP 求解器；不能写成“用 QUBO 直接求解堆叠问题”。
6. 后处理 polish 只能缩短已有可行路径，不能负责可行性逃逸。

## 8. 推荐写法

对外或论文草稿里建议这样表述：

> 当前最强的单一质量算法是 `qubo_scheduler_state_v5_wang_rescue_polish`。它以 compact Beam search 为主体，在风险层引入 Wang-inspired pair-QUBO rescue，并在找到 `N=0` 后用 QUBO 选择 suffix/window 做局部路径压缩。在 open DLTS 5x5/7x5 test-120 上，该算法解出 118/120，full-quality penalty 为 3873，成功解总步数 3692；相比 Wang BS-B，它少解 2 例，但在共同解出的 118 例上少 804 步，质量优势明显。Wang BS-B 应作为强可行性 fallback，而不是质量主算法。

如果需要强调内部 30 例：

> 对上港内部 30 个单贝位算例，默认主线仍是 `speed_v2` / `scheduler_state_v5`：它们已经达到 30/30，`speed_v2` 为 509 步、平均 2.237 秒。Wang rescue 在内部 30 上只带来 1 步级别收益，但运行成本显著上升，因此不应无条件作为内部默认算法。

## 9. 复现命令

open DLTS 当前最佳单一质量算法：

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_wang_rescue_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_wang_rescue_polish_test120_current.csv
```

内部 30 例默认主线：

```powershell
python scripts\benchmark_compact_beam_qubo_n0.py --profile speed_v2 --output docs\experiments\micro_optimized_speed_v2_full30.csv
```

Wang BS-B 可行性 fallback：

```powershell
python scripts\benchmark_classical_cpmp_baselines.py --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --solvers wang_bs_b --time-limit 10 --max-depth 200 --beam-width 100 --max-candidates-per-layer 64 --output docs\experiments\wang_bs_b_open_5x5_7x5_test120.csv
```

## 10. 后续最值得做的事

下一步不应再简单加大 QUBO 变量数或 supplement width。当前证据更支持以下方向：

1. 聚焦两个残余失败 case：`5x5_cv1_test_4` 和 `5x5_cv2_test_10`。
2. 对比 `wang_rescue_polish` 与 Wang BS-B 的前几步，提取能逃离 compact plateau 的临时 slack 动作。
3. 把 Wang 的早期逃逸行为转成 compact parent-chain 内的 bounded escape operator，而不是拼接外部路径。
4. 保留 full polish 作为质量主线，同时研究更安全的 polish budget，避免 selective-window 带来新失败。
5. 继续按 full-case quality 判断实验，不只看成功 case 的平均步数。

## 11. 最短结论

当前最好的单一质量算法是：

```text
qubo_scheduler_state_v5_wang_rescue_polish
```

它的定位是：

```text
compact Beam 主搜索
+ scheduler-state 风险控制
+ Wang-inspired rescue 候选补充
+ QUBO pair / frontier selection
+ QUBO suffix/window polish
```

它的边界也必须一起写清楚：

```text
质量最好，但不是 100% feasibility；
内部 30 默认仍用 speed_v2；
Wang BS-B 是 fallback，不是质量主线；
QUBO 是候选选择器，不是完整求解器。
```
