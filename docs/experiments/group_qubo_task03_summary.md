# Task 03：分组 QUBO 实验记录

本轮目标是验证“分组/保护式 QUBO”是否能在 `speed_v2` 主线基础上带来稳定收益。实验只看 `n=0` 分支，并先用 26 号贝位的敏感算例做筛选。

## 入口修正

独立脚本 `scripts/benchmark_compact_group_qubo_majority.py` 原本用程序内参数调用主求解器，但主求解器会先应用 `speed_v2` profile，导致脚本里的 `core_width`、`qubo_group_select` 等参数有被覆盖的风险。

本轮已修正为：

- 默认声明 `solver_profile=speed_v2`。
- 将脚本构造出的参数写入 `_profile_explicit_keys`。
- 新增测试锁定该行为。

## 敏感算例结果

算例集合：`A2_26`、`B3_26`、`C2_26`、`C5_26`。

| 方案 | 解出 | 成功算例总步数 | 平均时间(秒) | 失败算例 |
|---|---:|---:|---:|---|
| `speed_v2` 主线 | 4/4 | 112 | 4.338 | - |
| 主脚本 group supplement | 4/4 | 112 | 4.766 | - |
| majority，core=600 | 2/4 | 55 | 3.495 | `B3_26`, `C2_26` |
| majority，core=900, select=36, take=4 | 3/4 | 81 | 4.886 | `C2_26` |
| majority，core=1000, select=18, take=4 | 3/4 | 87 | 4.639 | `B3_26` |
| majority，core=1080, select=24, take=4 | 4/4 | 114 | 5.958 | - |
| majority，core=1080, select=12, take=4 | 4/4 | 114 | 5.910 | - |

## 结论

分组 QUBO 已经作为实验分支验证过，但暂不进入当前 `n=0` 高效主线。

原因：

- `group supplement` 能保持全解，但没有减少步数，时间从 4.338 秒升到 4.766 秒。
- 激进 `majority` 能在 `C5_26` 上偶发少一步，但会破坏 `B3_26`、`C2_26` 的可解性。
- 保守 `majority` 能恢复全解，但总步数从 112 增到 114，平均时间也明显增加。

当前主线仍应保留 `speed_v2`：状态级 QUBO + 后处理 QUBO。分组 QUBO 可以保留为“扩大 QUBO 影响”的反例证据：简单扩大 QUBO 控制范围并不会自然提升主线，反而会削弱稳定性。

## 相关输出

- `docs/experiments/group_probe_speed_v2_baseline_sensitive.csv`
- `docs/experiments/group_probe_supplement_sensitive.csv`
- `docs/experiments/group_probe_majority_sensitive.csv`
- `docs/experiments/group_probe_majority_core900_s36_t4_sensitive.csv`
- `docs/experiments/group_probe_majority_core1000_s18_t4_sensitive.csv`
- `docs/experiments/group_probe_majority_core1080_s24_t4_sensitive.csv`
- `docs/experiments/group_probe_majority_core1080_s12_t4_sensitive.csv`
