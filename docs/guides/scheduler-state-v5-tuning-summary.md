# scheduler_state v5 调参记录

日期：2026-06-12

本文记录 `scheduler_state_v3/v4` 之后的调参结论。

## v5 改动

`scheduler_state_v5` 保留 v4 的 initial plateau rescue，但增加一个在线保护条件：

- 如果实例是唯一优先级（`priority_duplication_ratio == 0`），且初始逆序数较低（`initial_n < 24`），不锁定 initial plateau rescue。
- 重复优先级实例不受这个保护条件限制，因此 `5x5_cv3_test_6` 这类低 N 分组实例仍可触发 rescue。

目标是保留 v4 在 7x5 上的解出率收益，同时去掉 5x5 唯一优先级低 N 实例上的无效锁定。

## 公开算例矩阵，3 秒限制

| 版本 | 范围 | 解出 | 总步数 | 平均时间秒 | 锁定数 | 锁定且解出 | 锁定但失败 |
|---|---|---:|---:|---:|---:|---:|---:|
| `scheduler_state_v3` current | 7x5 cv1 test+validation | 21/40 | 790 | 2.108 | 0 | 0 | 0 |
| `scheduler_state_v4` | 7x5 cv1 test+validation | 23/40 | 878 | 2.150 | 12 | 6 | 6 |
| `scheduler_state_v5` | 7x5 cv1 test+validation | 23/40 | 878 | 2.135 | 11 | 6 | 5 |
| `scheduler_state_v3` current | 7x5 cv2+cv3 test | 20/20 | 738 | 2.273 | 2 | 2 | 0 |
| `scheduler_state_v4` | 7x5 cv2+cv3 test | 20/20 | 733 | 2.251 | 2 | 2 | 0 |
| `scheduler_state_v5` | 7x5 cv2+cv3 test | 20/20 | 738 | 2.286 | 2 | 2 | 0 |
| `scheduler_state_v3` current | 5x5 cv1 test+validation | 20/40 | 575 | 0.944 | 0 | 0 | 0 |
| `scheduler_state_v4` | 5x5 cv1 test+validation | 20/40 | 574 | 1.000 | 6 | 0 | 6 |
| `scheduler_state_v5` | 5x5 cv1 test+validation | 20/40 | 575 | 0.949 | 0 | 0 | 0 |
| `scheduler_state_v4` | 5x5 cv2+cv3 test | 16/20 | 404 | 0.797 | 4 | 1 | 3 |
| `scheduler_state_v5` | 5x5 cv2+cv3 test | 16/20 | 404 | 0.790 | 4 | 1 | 3 |

## 解释

v4 说明：放开唯一优先级实例的 initial plateau rescue 可以额外解出两个 7x5 cv1 validation case。代价是 5x5 唯一优先级低 N 实例出现无效锁定。

v5 保留了 7x5 cv1 的有效收益，并移除了这类 5x5 无效锁定。因此 v5 比 v4 更干净。它没有减少额外解出两个 7x5 cv1 case 带来的总步数增加。

额外探测没有改变主结论：

- 不锁定的短 rescue 会丢掉 `7x5_cv1_validation_12` 和 `7x5_cv1_validation_17`。
- `scheduler_state_initial_plateau_window=4` 会锁住更多失败的 7x5 cv1 case，但没有新增解出。
- 将 `scheduler_state_rescue_cap_slack` 提到 12 或 16 会增加 rescue 层数，但抽样失败例没有新增解出。
- 在 v5 新增的长路径上做 window polish，没有压短 `7x5_cv1_validation_12` 或 `7x5_cv1_validation_17`。

## 非 adaptive 对齐复查

v5 与 `scheduler_state_v3` current 在共同解出的 21 个 7x5 cv1 case 上基本已经对齐：共同成功集只多 2 步，平均时间只多约 0.001 秒。v5 总步数从 790 增到 878，主要原因不是共同 case 劣化，而是额外解出了两个 case：

| 新增解出 case | v3 current 最终 N | v5 最终 N | v5 步数 |
|---|---:|---:|---:|
| `7x5_cv1_validation_12` | 25 | 0 | 41 |
| `7x5_cv1_validation_17` | 24 | 0 | 45 |

因此，“对齐非 adaptive”的主要结论是：v5 已经保住共同成功集，剩余总步数差异来自新增可行解本身。

## v6/v7 后处理探测

新增了两个实验 profile，但目前不推荐替代 v5：

- `scheduler_state_v6`：v5 搜索，7-stack 及以上使用 `qubo_windows_v2`，5-stack 回退 legacy polish。当前复测中 5x5 与 v5 对齐；7x5 没有稳定优于 v5。
- `scheduler_state_v7`：v5 搜索，加强 suffix polish，不启用 window repair。7x5 cv1 为 23/40、877 步、平均 2.200 秒；只比 v5 少 1 步，但比 v5 慢约 0.065 秒。

关键负结果：window repair 本身在本轮没有成功缩短路径，`post_polish_window_moves_saved=0`。少数步数波动来自 suffix polish 和搜索随机性，不足以作为默认主线证据。当前主推仍是 `scheduler_state_v5`。

## 复现命令

```powershell
python scripts\open_cpmp_benchmarks.py --families 7x5_cv1_test 7x5_cv1_validation --limit-per-family 20 --variant qubo_scheduler_state_v5 --time-limit 3 --no-auto-clone --output docs\experiments\scratch_state_v5_seq_7x5_cv1_40.csv
python scripts\open_cpmp_benchmarks.py --families 7x5_cv2_test 7x5_cv3_test --limit-per-family 10 --variant qubo_scheduler_state_v5 --time-limit 3 --no-auto-clone --output docs\experiments\scratch_state_v5_seq_7x5_cv23_20.csv
python scripts\open_cpmp_benchmarks.py --families 5x5_cv1_test 5x5_cv1_validation --limit-per-family 20 --variant qubo_scheduler_state_v5 --time-limit 3 --no-auto-clone --output docs\experiments\scratch_state_v5_seq_5x5_cv1_40.csv
python scripts\open_cpmp_benchmarks.py --families 5x5_cv2_test 5x5_cv3_test --limit-per-family 10 --variant qubo_scheduler_state_v5 --time-limit 3 --no-auto-clone --output docs\experiments\scratch_state_v5_seq_5x5_cv23_20.csv
```
