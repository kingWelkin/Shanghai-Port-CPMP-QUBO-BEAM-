"""QUBO 精炼阶段辅助（子问题变量选取与精炼指标报告）。"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .metrics import evaluate_g_trajectory, extract_bvx_from_var_index


def select_key_variables_shuffle(
    pool_vec: List[np.ndarray],
    sub_size: int,
    rng: np.random.Generator,
) -> List[int]:
    mat = np.asarray(pool_vec, dtype=np.float64)
    p = np.mean(mat, axis=0)
    uncertainty = np.abs(p - 0.5)
    n = mat.shape[1]
    k = min(sub_size, n)
    order = np.argsort(uncertainty)
    key = order[:k].tolist()
    if len(pool_vec) > 1:
        spread = float(np.max(uncertainty) - np.min(uncertainty[key]))
        if spread < 1e-9:
            key = rng.choice(n, size=k, replace=False).tolist()
    return key


def report_shuffle_metrics(
    problem,
    best: np.ndarray,
    meta: Dict,
    var_index: Dict,
) -> List[str]:
    W, H, N = problem.W, problem.H, problem.N
    T = int(meta["T_qubo"])
    b_vals, x_vals = extract_bvx_from_var_index(best, var_index)
    x_vals = {k: v for k, v in x_vals.items() if (k[0], k[1]) != (k[2], k[3])}
    traj = evaluate_g_trajectory(problem.bay.pri, W, H, N, T, b_vals, x_vals)
    moves = traj[-1][1] if traj else 0
    n_end = traj[-1][2] if traj else 0
    cross = sum(1 for i, j, k, l, n, t in x_vals if i != k)
    same = sum(1 for i, j, k, l, n, t in x_vals if i == k)
    label = "跨堆翻箱" if meta.get("mode") == "yard_reloc" else "同堆翻箱"
    lines = [
        f"模式: {meta.get('mode')}",
        f"N(0)={meta['n0']}",
        f"T_qubo={T}",
        f"{label}次数(末)={moves}",
        f"末态 N(Future)={n_end}",
        f"末态 G={moves + n_end}",
    ]
    if meta.get("mode") == "yard_reloc":
        lines.append(f"编码中跨堆动作数: {cross}  （同堆动作数: {same}，模型禁止 i=k）")
    lines.append("")
    return lines
