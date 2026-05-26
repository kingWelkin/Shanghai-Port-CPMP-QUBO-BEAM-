# 开发说明

## 源码布局

```text
src/greedy/
├── __main__.py
├── run_batch.py          # CLI 入口
├── input_loader.py       # 单文件输入加载
├── multi_bay_loader.py   # 多贝位合并
├── greedy_plan.py        # 贪心规划核心
├── greedy_moves.py       # 贪心动作应用/模拟
├── metrics.py            # N(Future)、G 等指标
├── greedy_report.py      # 贪心轨迹报告
├── report.py             # QUBO/通用报告
└── *_qubo.py / sa_solver.py
```

## 推荐开发命令

```powershell
python -m pip install -e .
python -m greedy --zone A1 --out-dir tmp/results
```

批量验证：

```powershell
python -m greedy --out-dir tmp/results
```

## 默认路径

默认路径在 `src/greedy/run_batch.py` 中定义：

```text
input:  data/input
output: data/results
```

`config/defaults.toml` 当前只是参数约定记录，代码尚未读取它。

## 当前需要优先注意的问题

### 包名

规范化后正式包名是：

```text
greedy
```

旧文档中的 `greedy_qubo` 已不再作为当前包名使用。

### QUBO 精炼

QUBO 路径不是默认稳定路径。继续开发前建议先处理：

- 贪心动作是带 `"cross"` 标签的 6 元组，而部分 QUBO 编码函数期望 5 元组。
- 贪心主约束使用 `1.5 * N(0)`，部分 QUBO meta 仍有旧比例。
- 全量变量枚举在当前案例规模上很重，需要裁剪候选动作或缩小窗口。

### 贪心策略

如果目标改为“尽可能达到全场良序”，需要重新设计停止策略。当前贪心会过滤掉让 `N` 暂时增加的动作，这能保证过程稳健，但也可能阻止必要的绕行动作。

## 建议补充

- 增加 `tests/`，覆盖输入解析、`N(Future)`、单步搬箱和 A1 smoke test。
- 将 `config/defaults.toml` 接入 CLI。
- 为 QUBO 精炼增加小规模 fixture，先在 `W<=4, H<=3` 的人工案例上验证。
