# Task 07：微优化实验记录

本轮目标是在不改变 `speed_v2` 搜索策略、不改变 `n=0` 结果分支的前提下，把主线速度拉回历史 2.2 秒档。

## profile 结论

以 `A2_26` 为代表样例做 cProfile，主要热点为：

- 后继生成 `_successors_with_n_cached`：约 4.5 秒累计时间。
- 候选排序 `_select_next_frontier`：约 1.8 秒累计时间。
- QUBO 选择本身：约 0.5 秒量级，不是本轮主要瓶颈。

因此本轮只做等价微优化，不改变 QUBO 策略。

## 保留的优化

1. 单堆 pop/push chunk 缓存

   将后继生成中的单堆出栈、入栈 byte chunk 构造改为缓存 helper：

   - `_cached_pop_chunk`
   - `_cached_push_chunk`

   successor 输出集合保持与参考实现一致。

2. chunk 元信息合并缓存

   新增 `_cached_chunk_meta(chunk, max_height)`，一次返回：

   - 当前堆高度
   - 当前堆的逆序贡献

   作用是减少每个 expanded state 上重复查询 height 和 stack_n 的缓存包装开销。

3. 默认排序 key 专用化

   `speed_v2` 固定使用 `n_then_depth`。新增 `_candidate_sort_key` / `_candidate_key_func`，在默认优先级下直接使用：

   ```text
   (n_value, depth, parent_code, code)
   ```

   其他优先级仍回退到原 `_candidate_rank`。

## 放弃的优化

尝试过用 `(source_chunk, dest_chunk)` 提前跳过重复移动对，但敏感 4 例变慢：

- 4/4 可解，112 步不变。
- 平均时间约 4.455 秒，差于保留方案。

因此该剪枝已撤回。

## full30 结果

输出文件：`docs/experiments/micro_optimized_speed_v2_full30.csv`

| 版本 | 解出 | 总步数 | 平均时间(秒) | 最大时间(秒) | 26号贝位总步数 | 26号贝位平均时间(秒) |
|---|---:|---:|---:|---:|---:|---:|
| 本轮微优化 `speed_v2` | 30/30 | 509 | 2.237 | 5.597 | 414 | 4.031 |
| 之前重跑 `redo_speed_v2_baseline_full30` | 30/30 | 509 | 2.741 | 6.275 | 414 | 4.911 |
| 历史 2.2 秒档 `compact_beam_qubo_speed_optimized_v2_full30_repeat` | 30/30 | 509 | 2.274 | 4.752 | 414 | 4.107 |

## 结论

Task 07 可以收口。当前主线保持：

- `n=0`：30/30。
- 总步数：509。
- 平均时间：2.237 秒。

后续严格消融和文档结论应以这个微优化后的 `speed_v2` 作为当前高效主线。
