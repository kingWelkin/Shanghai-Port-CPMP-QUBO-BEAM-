"""本地模拟退火子 QUBO 求解（dimod + neal）。"""

from __future__ import annotations

import numpy as np
from dimod import BinaryQuadraticModel
from neal import SimulatedAnnealingSampler

RANDOM_SEED = 42


def solve_qubo_sa(
    Q: np.ndarray,
    num_reads: int = 5000,
    sampler=None,
    seed: int | None = None,
    verbose: bool = True,
) -> tuple:
    if seed is None:
        seed = RANDOM_SEED

    if not np.allclose(Q, Q.T):
        raise ValueError("Q矩阵必须是对称矩阵")

    n = Q.shape[0]
    if Q.shape[1] != n:
        raise ValueError("Q矩阵必须是方阵")

    linear = {}
    quadratic = {}
    for i in range(n):
        linear[str(i)] = Q[i, i]
        for j in range(i + 1, n):
            if Q[i, j] != 0:
                quadratic[(str(i), str(j))] = Q[i, j]

    bqm = BinaryQuadraticModel(linear, quadratic, "BINARY")

    if sampler is None:
        sampler = SimulatedAnnealingSampler()

    if verbose:
        print(f"开始求解QUBO问题，变量数量: {n}, 运行次数: {num_reads}, 随机种子: {seed}")

    try:
        sampleset = sampler.sample(bqm, num_reads=num_reads, seed=seed)
    except TypeError:
        try:
            sampleset = sampler.sample(bqm, num_reads=num_reads)
        except TypeError:
            try:
                sampleset = sampler.sample(bqm, seed=seed)
            except TypeError:
                sampleset = sampler.sample(bqm)

    best_sample = sampleset.first.sample
    best_solution = np.zeros(n, dtype=int)

    for i in range(n):
        val = best_sample.get(str(i), 0)
        if val == -1:
            val = 0
        elif val not in [0, 1]:
            val = 1 if val > 0 else 0
        best_solution[i] = int(val)

    if verbose:
        best_energy = sampleset.first.energy
        print(f"求解完成，最佳能量值: {best_energy:.6f}")
        print(f"最佳解中1的个数: {np.sum(best_solution)}")

    return sampleset, best_solution


def solve_subqubo_local(
    subQ: np.ndarray, seed: int = 42, *, num_reads: int = 500
) -> np.ndarray:
    _, best_solution = solve_qubo_sa(subQ, num_reads=num_reads, seed=seed, verbose=False)
    return best_solution.astype(np.int8)
