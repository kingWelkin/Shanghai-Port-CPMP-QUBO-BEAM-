# greedy_qubo（多贝位合并 + 贪心搬箱）

在**同一箱区**内把多个贝位（默认 **01 + 26**）合成**一场**堆场：允许**跨堆、跨贝**搬**顶箱**，**禁止同堆内翻箱**。先用确定性**贪心**生成可行搬箱序列（快），可选再用**短窗口 QUBO + 模拟退火**做小幅精炼。

与 `main_stack_shuffle.py`（同堆翻箱、不跨贝）是不同问题，不要混用。

---

## 1. 核心在做什么（两阶段）

| 阶段 | 模块 | 做什么 |
|------|------|--------|
| **① 贪心规划** | `greedy_plan.py` / `greedy_moves.py` | 从初始布局出发，反复选一步「跨堆搬顶箱」，直到全场各堆良序或无路可走 |
| **② QUBO 精炼（可选）** | `solve.py` + `yard_reloc_qubo.py` | 把贪心后半段放进时间窗口，建 QUBO，用本地 SA 尝试降低能量（默认**不开启**） |

默认批量跑**只做阶段 ①**，几秒到几十秒一个箱区。

---

## 2. 指标与约束（读结果前必看）

### 逆序箱数 N(Future)

自**第 2 层**向上：找**第一个**「上层关序号 **>** 下层关序号」的层 j，把**该层及以上**所有箱计入逆序。比较用**关序号**（seq），不是秩。

**良序**：下层关序号 > 上层（上层关序号更小）。

### 组合指标 G

```text
G = 已执行搬箱步数 + N(Future)
```

约束（全程任意中间态都要满足）：

```text
G ≤ floor(1.5 × N(0))
```

- **N(0)**：初始全场逆序箱数（合并多场后按**合并后的 W 堆**一起算）
- **不是**要求 G 或 N 必须比初始**减少**，只要求**不超过** `1.5×N(0)`
- 例：N(0)=29 → G 最多 **43**

### 允许 / 禁止的动作

- **允许**：从堆 i 搬走**最顶**一箱，放到堆 k 的**顶**（`i≠k`，可跨贝，如 `01-A` → `26-3`）
- **禁止**：同堆内调整层位（那是 `main_stack_shuffle` 的事）

### 贪心如何选一步（`greedy_plan.plan_greedy_until_terminal`）

1. 枚举所有合法 **cross** 搬顶箱候选（源堆有箱、目标可接顶箱）。
2. 模拟一步后算新的 `G' = 步数 + N(Future)`；若 `G' > G_cap` 则丢弃。
3. 若布局与历史重复则丢弃（防循环）。
4. 在剩余候选里：**优先 N(Future) 不升高**；再让 **G 尽量小**；多贝时还会用 `drain_bay` / `fill_bay`（初始各贝 N(0) 高→低）做 tie-break，倾向从逆序多的贝往外搬、往逆序少的贝收。
5. 重复直到全场良序、无合法步、或达到 `--max-greedy-moves` 上限。

---

## 3. 输入数据怎么准备

### 单文件格式

与 `ORAL/input/输入说明.txt` 一致，每行：

```text
位置编码 关序号
```

例：`A101A1 1001`（箱区 A1、贝 01、排 A、层 1、关序号 1001）。

### 默认模式：多贝合并

需要在 **`ORAL/input/`** 下**同时存在**同一箱区的：

- `{ZONE}_01_input.txt`
- `{ZONE}_26_input.txt`

例如 `A1_01_input.txt` + `A1_26_input.txt` → 合并为案例 **`A1_01_26_paired`**：

- 全局堆号：贝 01 的 10 堆 → 堆 1–10，贝 26 的 10 堆 → 堆 11–20（`W=20`）
- 全场共用一套关序号秩（按 seq_id 统一排序）

### 旧模式：单文件不合并

加 `--per-file` 时，对每个 `*_input.txt` 单独跑，**不**跨贝合并（`W=10`）。

---

## 4. 怎么运行（具体操作）

### 环境

- 工作目录应为 **`ORAL`**（包名 `greedy_qubo` 挂在 ORAL 下）
- 依赖：`numpy`、`dimod`、`neal` 等（与主工程 `solver.py` 相同）
- 可用 `run.bat`（默认 `ising-env` 的 Python）

### 进入目录

```powershell
cd d:\brp-decom\ORAL
```

### 最常用：默认批量（01+26 合并，仅贪心）

```powershell
python -m greedy_qubo
```

等价于扫描 `input/` 里所有能配对的箱区（A1、A2…），每个输出到 `greedy_qubo/results/result_{ZONE}_01_26_paired.txt`，并写 `greedy_qubo/results/SUMMARY.txt`。

### 只跑一个箱区

```powershell
python -m greedy_qubo --zone A1
```

### 指定合并的贝位后缀

```powershell
python -m greedy_qubo --zone B2 --bays 01,26
```

（目录里必须有 `B2_01_input.txt` 与 `B2_26_input.txt`。）

### 开启 QUBO 精炼（慢一些）

```powershell
python -m greedy_qubo --zone A1 --refine --iters 3 --sub-size 40 --sa-reads 80
```

- 若贪心**已达终态良序**，且未加 `--refine-always`，会**跳过** QUBO 并打印 `[QUBO] 跳过：贪心已达终态良序`
- `--refine-always`：即使已良序也做精炼

### 旧模式：每个 input 单独跑

```powershell
python -m greedy_qubo --per-file
python -m greedy_qubo --per-file --glob "A1_*_input.txt"
```

输出名为 `result_A1_01_input.txt` 等。

### Windows 批处理

```powershell
cd d:\brp-decom\ORAL\greedy_qubo
.\run.bat
.\run.bat --zone A1
```

`run.bat` 会 `cd` 到 `ORAL` 再执行 `python -m greedy_qubo`。

### 常用参数一览

| 参数 | 默认 | 含义 |
|------|------|------|
| `--input-dir` | `ORAL/input` | 输入 txt 目录 |
| `--out-dir` | `greedy_qubo/results` | 结果目录 |
| `--bays` | `01,26` | 合并的贝位后缀 |
| `--zone` | （空=全部） | 只跑指定箱区 |
| `--per-file` | 关 | 不合并，单文件模式 |
| `--refine` | 关 | 贪心后再 QUBO |
| `--max-greedy-moves` | 0（不限） | 贪心最大步数 |
| `--skip-if-moves-above` | 0（不跳过） | 步数超过则 SKIP |
| `--timing` | 关 | 打印 QUBO 建模耗时 |

---

## 5. 输出文件怎么看

成功时：`results/result_{案例名}.txt`

- 头部：`greedy_moves`、`terminal_sorted`、`N(0)`、`G_end`、`G_cap`、`g_ok`
- 中间：各堆 N/G 变化、是否满足 `G ≤ G_cap`
- 末尾：初始/末态堆表、逐步 `MOVE` 列表（跨贝会标 `01-x → 26-y`）

失败时：同名的 `result_{案例名}_ERROR.txt`（完整 traceback）。

批量结束看：`results/SUMMARY.txt`（每案例一行 OK / ERR / SKIP）。

---

## 6. 代码结构（改逻辑时找哪里）

```text
greedy_qubo/
  run_batch.py / __main__.py   # 入口
  solve.py / refine_helpers.py # 贪心 + 可选 QUBO 精炼
  greedy_plan.py / greedy_moves.py / greedy_report.py
  multi_bay_loader.py / input_loader.py / problem_types.py
  metrics.py / inversion_metric.py
  yard_reloc_qubo.py / qubo_accum.py / sa_solver.py / full_encode.py
  cross_stack_reloc.py / in_stack_shuffle.py / report.py / utils.py
  results/                     # 默认输出
```

所有逻辑已合并在本目录，不再依赖单独的 `qubo_method` 包。

---

## 7. 与其它 ORAL 脚本的区别

| 脚本 | 动作范围 | 典型用途 |
|------|----------|----------|
| **greedy_qubo** | 跨堆/跨贝搬顶箱 | 整场堆场整理，约束 G≤1.5·N(0) |
| **main_stack_shuffle** | 仅同堆内翻层 | 单堆排序，不提箱、不跨堆 |
| **empty_stack** | 排空/配对贝位等 | 另一类空贝 QUBO 流程 |

---

## 8. 快速自检

```powershell
cd d:\brp-decom\ORAL
python -m greedy_qubo --zone A1
```

终端应出现 `=== A1_01_26_paired ===`、各贝 `N(0)`、贪心步数与 `g_ok=True/False`，并在 `greedy_qubo/results/` 生成 `result_A1_01_26_paired.txt`。

若报「未找到可合并案例」，检查 `input/A1_01_input.txt` 与 `input/A1_26_input.txt` 是否都存在。
