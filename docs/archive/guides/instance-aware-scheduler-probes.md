# 实例感知调度器试验记录

> [!IMPORTANT]
> Historical document. It does not define the current default. The promoted mainline is [v4.3](../../guides/v43-mainline-algorithm.md), validated at `1681/1681` with `105530` moves.

本文记录 2026-06-12 对三条后续路线的实现和小矩阵结果：

- 实例感知 profile scheduler；
- 轻量 rescue QUBO；
- 分桶 bounded top-k 候选裁剪。

结论先写清楚：`scheduler_v1` 是当前最值得继续扩展的开放算例主线候选，但不是全局胜利。signal-aware gate 在 `7x5_cv2/cv3` 前 10 例保持 10/10，并比轻量 rescue 更快；在 `7x5_cv1` 前 10 例只有 6/10，说明 hard unique-priority case 仍需要单独机制。`scheduler_topk_v1` 已有可复现入口，但本轮 `rank_truncated=0`，尚未证明真实加速收益。

## 1. 新增 profile

### `light_rescue_qubo_v1`

这是 `rescue_qubo_v1` 的保守版。相比完整 rescue，它降低了干预强度：

- `rescue_max_layers`: `16 -> 8`
- `rescue_beam_width/core_width`: `1350/1215 -> 1250/1125`
- `rescue_escalate_after`: `2 -> 4`
- `rescue_max_qubo_vars`: `96 -> 80`
- `rescue_qubo_supplement_width`: `48 -> 32`

定位：在 `speed_v2` 基础上只对平台硬例做局部救援，避免全局放大 QUBO。

### `scheduler_v1`

这是实例感知调度器：

- 默认保持 `speed_v2` 的 enhanced selection；
- 出现 plateau、候选数压力和 cap slack 紧张时触发 light rescue；
- adaptive 起步不读取算例名，而是检查重复优先级比例、`best_n / initial_n`、候选压力、父节点集中度和 plateau/slack；
- 重复优先级比例超过阈值时禁用 adaptive 起步；唯一优先级且高 `N` 的 7-stack 层仍可使用 adaptive frontier。

关键保护从旧版 `scheduler_adaptive_cv1_only=True` 改为结构信号：`scheduler_adaptive_max_duplication_ratio=0.0`、`scheduler_adaptive_min_best_n_ratio=0.45`，并在 plateau 且 slack 紧张时禁用 adaptive。本轮 cv2 小矩阵显示，直接把 adaptive 用到重复优先级 hard case 会救出 `7x5_cv2_test_9`，但会误伤 `7x5_cv2_test_2`；signal-aware gate 后恢复为 10/10。

### `scheduler_topk_v1`

这是候选处理加速入口：

- `rank_bucket_topk=64`
- `rank_bucket_key=n_depth_parent`
- `rank_bucket_topk_high_n=18`
- `rank_bucket_topk_min_candidates=10000`

它只在候选数足够大且 best `N` 仍较高时启用分桶 top-k；低 `N` 末段自动回到完整排序，避免切掉 rescue 所需路径。

## 2. 7x5 前 10 小矩阵

范围：`7x5_cv1/cv2/cv3_test` 前 10 个 case，`time_limit=5`。

| family | 变体 | 解出 | 成功算例总步数 | 总时间 sec | 平均时间 sec | QUBO calls | rescue layers | rank truncated |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `7x5_cv1_test` | `scheduler_v1` signal-aware | 6/10 | 222 | 29.580 | 2.958 | 306 | 70 | 0 |
| `7x5_cv2_test` | `scheduler_v1` signal-aware | 10/10 | 381 | 29.254 | 2.925 | 324 | 62 | 0 |
| `7x5_cv3_test` | `scheduler_v1` signal-aware | 10/10 | 354 | 24.863 | 2.486 | 321 | 56 | 0 |

`7x5_cv2_test` 前 10 的同口径对照：

| 变体 | 解出 | 成功算例总步数 | 总时间 sec | 平均时间 sec | QUBO calls | rescue layers | rank truncated |
|---|---:|---:|---:|---:|---:|---:|---:|
| `speed_v2` | 9/10 | 330 | 36.421 | 3.642 | 218 | 0 | 0 |
| `light_rescue_qubo_v1` | 10/10 | 379 | 38.526 | 3.853 | 324 | 59 | 0 |
| `scheduler_v1` signal-aware | 10/10 | 381 | 29.254 | 2.925 | 324 | 62 | 0 |
| `scheduler_topk_v1` signal-aware | 10/10 | 381 | 32.799 | 3.280 | 327 | 60 | 0 |

解释：

- `speed_v2` 仍是内部 30 主线，但在 `7x5_cv2_test_9` 失败。
- `light_rescue_qubo_v1` 救出 hard case，但步数 gap 增大。
- `scheduler_v1` 通过 signal-aware gate 避免 adaptive 误伤重复优先级 hard case，在 `cv2/cv3` 前 10 中保住 10/10，但 `cv1` 只有 6/10。
- `scheduler_topk_v1` 的 `rank_truncated=0`，说明本轮门控 top-k 没有真正触发；其速度数字不能归因于候选裁剪收益。

## 3. 复现命令

```powershell
python scripts\open_cpmp_benchmarks.py --families 7x5_cv2_test --limit-per-family 10 --variant qubo --solver-profile speed_v2 --time-limit 5 --no-auto-clone --output docs\experiments\scratch_7x5_cv2_10_speed_v2.csv
python scripts\open_cpmp_benchmarks.py --families 7x5_cv2_test --limit-per-family 10 --variant qubo_light_rescue --time-limit 5 --no-auto-clone --output docs\experiments\scratch_7x5_cv2_10_light_rescue.csv
python scripts\open_cpmp_benchmarks.py --families 7x5_cv1_test --limit-per-family 10 --variant qubo_scheduler --time-limit 5 --no-auto-clone --output docs\experiments\scratch_signal_7x5_cv1_scheduler.csv
python scripts\open_cpmp_benchmarks.py --families 7x5_cv2_test --limit-per-family 10 --variant qubo_scheduler --time-limit 5 --no-auto-clone --output docs\experiments\scratch_signal_7x5_cv2_scheduler.csv
python scripts\open_cpmp_benchmarks.py --families 7x5_cv3_test --limit-per-family 10 --variant qubo_scheduler --time-limit 5 --no-auto-clone --output docs\experiments\scratch_signal_7x5_cv3_scheduler.csv
python scripts\open_cpmp_benchmarks.py --families 7x5_cv2_test --limit-per-family 10 --variant qubo_scheduler_topk --time-limit 5 --no-auto-clone --output docs\experiments\scratch_signal_7x5_cv2_scheduler_topk.csv
```

## 4. 当前判断

后续应优先扩大 `scheduler_v1` 的外部矩阵，并重点解剖 `7x5_cv1` 失败例：

- `5x5` cv1/cv2/cv3；
- `7x5` cv1/cv2/cv3；
- 同时对比 `speed_v2`、`scheduler_v1`、`classic_noqubo` time-matched。

`scheduler_topk_v1` 暂时不作为推荐算法。下一步若继续候选处理加速，应做真正生成期的 bounded heap，而不是只在生成后分桶筛选。

## 5. State scheduler v2 补充

新增 `scheduler_state_v2`，入口为：

```powershell
python scripts\open_cpmp_benchmarks.py --variant qubo_scheduler_state ...
```

它不再按算例名称或单纯问题大小决定是否启用 adaptive，而是在每层根据求解状态选择动作：

- `PROTECT`：默认路径，使用 `speed_v2` enhanced frontier，保护内部 30 和末段稀有路径。
- `CONTRACT`：当候选压力处于中等高压、frontier 未坍缩、top-N 不确定性较低时，启用 adaptive frontier。
- `RESCUE`：当 `best_n` 平台、depth slack 紧张时，触发局部 rescue。

关键保护：

- `scheduler_state_max_contract_pressure=40.0`：候选压力极端高时不收窄。这个规则修复了 `C2_26_single` 被误收窄后无法归零的问题。
- `scheduler_state_progress_delta=0`：允许 plateau 但候选压力适中的层继续 adaptive 推进。这个规则把 `7x5_cv1` 40 例从 18/40 拉回 21/40。

本轮 `time_limit=3` 关键矩阵：

| 范围 | `scheduler_state_v2` 结果 | 说明 |
|---|---:|---|
| 内部 30 | 30/30, 509 moves, avg 1.826s | 与 `speed_v2` 质量持平，避免 adaptive 误伤 26 贝位 |
| 5x5 cv1 test+validation | 20/40, 575 moves, avg 1.006s | 比旧 adaptive 多 1 例，但 gap 变大 |
| 5x5 cv2+cv3 test | 16/20, 404 moves, avg 0.832s | 与轻量路线基本持平 |
| 7x5 cv1 test+validation | 21/40, 791 moves, avg 2.179s | 回到 adaptive_controller 的解出率，但更慢一点 |
| 7x5 cv2+cv3 test | 19/20, 682 moves, avg 2.273s | 仍有 `7x5_cv2_test_2` 失败 |

当前判断：`scheduler_state_v2` 是比 `adaptive_controller_v1` 更安全的状态闭环候选，因为它保住内部 30/30；但它还没有同时吃下 `7x5_cv2_test_2` 和 `7x5_cv2_test_9`。后续若继续调，应重点研究 repeated-priority case 中何时需要更早保护，而不是继续按 family 名称分流。

## 6. State scheduler v3：重复优先级初始平台锁

`scheduler_state_v3` 是在 v2 上继续增强的规则版。核心变化不是按 family 或尺寸分流，而是识别一种在线状态：

- 优先级存在明显重复：`scheduler_state_initial_plateau_min_duplication=0.10`；
- `best_n` 在初始高位长期不下降：`scheduler_state_initial_plateau_window=5`，`scheduler_state_initial_plateau_min_ratio=0.98`；
- 触发后将该算例锁到 `rescue_qubo` 选择器：`scheduler_state_initial_plateau_action=rescue`，`scheduler_state_initial_plateau_lock=True`。

这个规则来自 `7x5_cv2_test_2` 和 `7x5_cv2_test_9` 的冲突：前者需要接近 `light_rescue_qubo_v1` 的保守选择，后者需要保留 state-contract。v3 用“重复优先级 + 初始高位平台”识别前者；`test_9` 不触发锁定，仍走 v2 的状态收窄。

负结果也保留结论：

- `failed_contract_cooldown=1` 会伤害连续收窄铺路的 hard case，不作为 profile 默认。
- post-rescue 后直接 `protect` 或 `blend` 不足以救 `test_2`，且会伤害内部唯一优先级算例，因此 v3 默认关闭 post-rescue guard。
- `blend` 的并集回填可以保留 QUBO 候选，但不能替代 `rescue_qubo` 选择器；`test_2` 的收益来自锁到 rescue-profile，而不是单纯加宽。

顺序复跑的公开 `time_limit=3` 矩阵如下：

| 范围 | `scheduler_state_v3` 结果 | 相对 v2 |
|---|---:|---|
| 5x5 cv1 test+validation | 20/40, 575 moves, avg 0.963s | 解出数和步数持平，略快 |
| 5x5 cv2+cv3 test | 16/20, 404 moves, avg 0.817s | 解出数和步数持平，略快 |
| 7x5 cv1 test+validation | 22/40, 837 moves, avg 2.140s | 多解 1 例，新增解约 46 步 |
| 7x5 cv2+cv3 test | 20/20, 733 moves, avg 2.237s | 多解 `7x5_cv2_test_2`，同时保住 `test_9` |

内部 30 的口径需要单独说明：在当前工作区按 `time_limit=3` 硬限复跑时，`speed_v2`、`scheduler_state_v2` 和 `scheduler_state_v3` 都不能稳定复现历史 30/30，因此不能把旧的内部 30/30 数字与本节公开 3 秒矩阵混在同一个结论里。v3 的新增规则带有重复优先级门控，理论上不会按唯一优先级内部算例触发，但内部主线仍应使用原先确认过的内部 benchmark 口径复核。
