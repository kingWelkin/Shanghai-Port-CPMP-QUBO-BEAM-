# 运行说明

## 环境

项目使用 Python 3.10+。依赖在根目录 `pyproject.toml` 中声明：

- `numpy`
- `dimod`
- `dwave-neal`

安装：

```powershell
python -m pip install -e .
```

如果不想安装包，可以临时设置 `PYTHONPATH`：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m greedy --zone A1
```

Windows 下也可以直接使用：

```powershell
.\run.bat --zone A1
```

`run.bat` 会自动把 `src` 加入 `PYTHONPATH`。

## 默认批量运行

```powershell
python -m greedy
```

默认行为：

- 输入目录：`data/input`
- 输出目录：`data/results`
- 合并贝位：`01,26`
- 模式：多贝位合并，仅贪心

运行时会扫描所有可配对箱区，例如：

```text
A1_01_input.txt + A1_26_input.txt -> A1_01_26_paired
B3_01_input.txt + B3_26_input.txt -> B3_01_26_paired
```

## 只跑一个箱区

```powershell
python -m greedy --zone A1
```

## 指定贝位

```powershell
python -m greedy --zone B2 --bays 01,26
```

贝位参数是输入文件名中的中间段，程序会查找：

```text
data/input/B2_01_input.txt
data/input/B2_26_input.txt
```

## 单文件模式

旧模式不合并贝位，每个 input 文件单独跑：

```powershell
python -m greedy --per-file
python -m greedy --per-file --glob "A1_*_input.txt"
```

## 输出目录

默认写入 `data/results`：

```text
data/results/result_A1_01_26_paired.txt
data/results/SUMMARY.txt
```

可以覆盖：

```powershell
python -m greedy --zone A1 --out-dir tmp/results
```

## QUBO 精炼

命令保留：

```powershell
python -m greedy --zone A1 --refine
```

但当前建议先把它视为实验路径。原因：

- 当前贪心主路径已经能快速生成可解释方案。
- QUBO 变量规模对 `W=20, H=6, N≈60` 的案例很大。
- 精炼路径需要继续校准动作编码和约束口径。
