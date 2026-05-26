"""两阶段：① 贪心可行解（快） ② 可选短窗口 QUBO 精炼。"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .metrics import n0_from_problem

from .greedy_plan import (
    greedy_plan_stats,
    plan_greedy_until_terminal,
    terminal_sort_reached,
)
from .greedy_report import format_greedy_placement_report, replay_moves_to_stacks


def run_greedy_phase(
    problem,
    *,
    seed: int = 42,
    greedy_max_moves: int | None = None,
    verbose: bool = True,
) -> Tuple[List, bool, Dict]:
    """仅贪心规划 + 重放校验，不建 QUBO。"""
    t0 = time.perf_counter()
    planned = plan_greedy_until_terminal(
        problem,
        seed=seed,
        max_moves=greedy_max_moves,
        verbose=verbose,
    )
    sorted_ok = terminal_sort_reached(problem, planned)
    stats = greedy_plan_stats(problem, planned)
    elapsed = time.perf_counter() - t0
    layout = (
        "multi_bay"
        if getattr(problem, "combined_bays", None)
        else "single_bay"
    )
    meta = {
        "mode": "multi_bay_reloc" if layout == "multi_bay" else "greedy_yard_reloc",
        "layout": layout,
        "greedy_moves": len(planned),
        "terminal_sorted": sorted_ok,
        "T_report": max(2, len(planned) + 2),
        "n0": n0_from_problem(problem),
        "phase": "greedy_only",
        "greedy_sec": round(elapsed, 4),
        **stats,
    }
    if verbose:
        print(
            f"  N {stats['n_future_init']}→{stats['n_future_end']}  "
            f"G {stats['g_init']}→{stats['g_end']}  "
            f"上限 {stats['g_cap']} 合规={stats['g_ok']}"
        )
    if verbose:
        print(
            f"  [贪心] {elapsed:.3f}s  步数={len(planned)}  终态良序={sorted_ok}"
        )
    return planned, sorted_ok, meta


def run_qubo_refine_phase(
    problem,
    planned: List,
    meta: Dict,
    *,
    seed: int = 42,
    sub_size: int = 40,
    iters: int = 3,
    penalty: float = 30.0,
    reloc_scale: float = 0.3,
    stack_order_scale: float = 5.0,
    terminal_stack_scale: float = 50.0,
    sa_reads: int = 80,
    verbose_iter: bool = True,
    verbose_build: bool = False,
) -> Tuple[np.ndarray, float, Dict, int, Dict, Dict]:
    """在 T_cap 短窗口上建 QUBO，以贪心后缀为初解做 SA。"""
    from .metrics import effective_move_cap
    from .sa_solver import solve_subqubo_local
    from .yard_reloc_qubo import build_yard_reloc_spec

    from .refine_helpers import select_key_variables_shuffle

    from .full_encode import (
        construct_full_yard_solution,
        count_feasibility_violations,
        flatten_full_yard_solution,
        split_moves_for_qubo_window,
    )

    _, T_cap = effective_move_cap(problem)
    prefix, suffix = split_moves_for_qubo_window(planned, T_cap)
    T_q = max(2, len(suffix) + 2)
    T_q = min(T_q, T_cap)

    t0 = time.perf_counter()
    spec = build_yard_reloc_spec(
        problem,
        penalty=penalty,
        reloc_scale=reloc_scale,
        stack_order_scale=stack_order_scale,
        terminal_stack_order_scale=terminal_stack_scale,
        time_horizon=T_q,
        auto_T=False,
        clip_time_to_cap=True,
        verbose_build=verbose_build,
    )
    build_sec = time.perf_counter() - t0

    if prefix:
        init_stacks = replay_moves_to_stacks(problem, prefix)
    else:
        init_stacks = None

    var_index = spec.var_index
    n_var = spec.n_var
    meta = dict(meta)
    meta.update(spec.meta)
    meta["T_qubo"] = T_q
    meta["T_cap"] = T_cap
    meta["qubo_window_moves"] = len(suffix)
    meta["qubo_prefix_moves"] = len(prefix)
    meta["phase"] = "greedy+qubo"
    meta["qubo_build_sec"] = round(build_sec, 3)

    sol = construct_full_yard_solution(
        problem, suffix, T=T_q, initial_stacks=init_stacks
    )
    base_vec = flatten_full_yard_solution(sol, var_index=var_index, n_var=n_var)
    viol = count_feasibility_violations(problem, sol, T=T_q)
    base_e = float(spec.energy(base_vec))

    if verbose_iter:
        print(
            f"  [QUBO建模] {build_sec:.1f}s  T={T_q} 窗口步={len(suffix)} "
            f"(总贪心 {meta.get('greedy_moves')} 步，前缀 {len(prefix)} 已重放)  "
            f"初解E={base_e:.2f}"
        )
        if viol:
            print(f"  窗口初解违反: {viol}")

    best = base_vec.copy()
    best_e = base_e
    rng = np.random.default_rng(seed)
    n_accept = 0
    t_sa = time.perf_counter()

    for it in range(1, max(0, int(iters)) + 1):
        key_vars = select_key_variables_shuffle([best], sub_size, rng)
        candidate = best.copy()
        subQ = spec.subqubo(key_vars, candidate)
        sub_sol = solve_subqubo_local(subQ, seed=int(seed + it), num_reads=sa_reads)
        for pos, gi in enumerate(key_vars):
            candidate[gi] = int(sub_sol[pos])
        e_new = float(spec.energy(candidate))
        prev_best = best_e
        accepted = e_new <= prev_best + 1e-9
        improved = e_new < prev_best - 1e-9
        if accepted:
            if improved:
                n_accept += 1
            best = candidate
            best_e = e_new
        if verbose_iter:
            tag = "↓" if improved else ("=" if accepted else "×")
            print(
                f"  iter {it:2d}: {e_new:.2f} → best {best_e:.2f} {tag}  |keys|={len(key_vars)}"
            )

    meta["qubo_sa_sec"] = round(time.perf_counter() - t_sa, 3)
    meta["n_accept"] = n_accept
    meta["init_violations"] = viol
    meta["init_energy"] = base_e
    meta["refined"] = iters > 0
    return best, best_e, meta, n_var, var_index, viol


def run_greedy_yard_solve(
    problem,
    *,
    refine: bool = False,
    sub_size: int = 40,
    iters: int = 3,
    seed: int = 42,
    penalty: float = 30.0,
    reloc_scale: float = 0.3,
    stack_order_scale: float = 5.0,
    terminal_stack_scale: float = 50.0,
    sa_reads: int = 80,
    verbose_iter: bool = True,
    verbose_build: bool = False,
    greedy_max_moves: int | None = None,
    refine_if_unsorted: bool = True,
) -> Tuple[Optional[np.ndarray], float, Dict, int, Dict, Dict, List]:
    """
    默认仅贪心（快）。``refine=True`` 时在 T_cap 短窗口上做 QUBO 改善。
    ``refine_if_unsorted=True`` 且已良序时跳过 QUBO。
    返回 ``planned`` 供报告使用。
    """
    planned, sorted_ok, meta = run_greedy_phase(
        problem,
        seed=seed,
        greedy_max_moves=greedy_max_moves,
        verbose=verbose_iter,
    )

    do_refine = bool(refine) and (not refine_if_unsorted or not sorted_ok)
    if not do_refine:
        meta["refined"] = False
        meta["best_energy"] = 0.0
        if verbose_iter and refine and sorted_ok:
            print("  [QUBO] 跳过：贪心已达终态良序")
        return None, 0.0, meta, 0, {}, {}, planned

    best, best_e, meta, n_var, var_index, viol = run_qubo_refine_phase(
        problem,
        planned,
        meta,
        seed=seed,
        sub_size=sub_size,
        iters=iters,
        penalty=penalty,
        reloc_scale=reloc_scale,
        stack_order_scale=stack_order_scale,
        terminal_stack_scale=terminal_stack_scale,
        sa_reads=sa_reads,
        verbose_iter=verbose_iter,
        verbose_build=verbose_build,
    )
    meta["best_energy"] = best_e
    return best, best_e, meta, n_var, var_index, viol, planned


def report_greedy_metrics(
    problem,
    best: Optional[np.ndarray],
    meta: Dict,
    var_index: Dict,
    init_viol: Dict,
) -> List[str]:
    lines = [
        f"模式: {meta.get('mode')}",
        f"贪心步数={meta.get('greedy_moves', '?')}",
        f"贪心耗时={meta.get('greedy_sec', '?')}s",
        f"贪心终态良序={meta.get('terminal_sorted', '?')}",
    ]
    if "n_future_end" in meta:
        lines.append(
            f"N(Future) {meta.get('n_future_init')}→{meta.get('n_future_end')}"
        )
        lines.append(
            f"G=翻箱+N  {meta.get('g_init')}→{meta.get('g_end')}  "
            f"须≤{meta.get('g_cap')}(=1.5×N0)  合规={meta.get('g_ok')}"
        )
    if meta.get("refined"):
        lines.extend(
            [
                f"QUBO建模={meta.get('qubo_build_sec')}s  SA={meta.get('qubo_sa_sec')}s",
                f"T_qubo={meta.get('T_qubo')} 窗口步={meta.get('qubo_window_moves')} "
                f"前缀={meta.get('qubo_prefix_moves')}",
                f"初解违反={init_viol or '无'}",
            ]
        )
        if best is not None and var_index:
            from .refine_helpers import report_shuffle_metrics

            lines.extend(report_shuffle_metrics(problem, best, meta, var_index))
    else:
        lines.append("QUBO精炼: 未执行（默认仅贪心）")
    lines.append("")
    return lines
