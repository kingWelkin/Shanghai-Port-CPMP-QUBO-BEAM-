"""Benchmark non-neural classical CPMP baselines on open CPMP instances."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from container_reshuffle.core import Instance, Solution, apply_move, create_initial_state, validate_instance, validate_solution  # noqa: E402
from container_reshuffle.core import future_disorder_count_stacks  # noqa: E402
from container_reshuffle.features import disorder_box_count  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig, failure_solution  # noqa: E402
from container_reshuffle.solvers.beam import BeamSearchSolver  # noqa: E402
from container_reshuffle.solvers.bs_frg import BsFrgSolver  # noqa: E402
from container_reshuffle.solvers.classical_cpmp import (  # noqa: E402
    BrkgaCpmpSolver,
    LpfhMultiHeuristicSolver,
    TargetGuidedBeamSolver,
    WangBsBabyMovesSolver,
    WangBsGiantMovesSolver,
)
from container_reshuffle.solvers.greedy import MultiRuleGreedySolver  # noqa: E402


FIELDS = [
    "family",
    "case",
    "solver",
    "moves",
    "final_n",
    "reference_moves",
    "move_gap",
    "runtime_sec",
    "expanded_states",
    "generated_states",
    "visited_states",
    "goal_validation_ok",
    "validation_msg",
    "status",
]

PRIORITY_GROUP_SOLVERS = {"wang_bs_g", "wang_bs_b", "bs_frg"}


def build_solver(name: str):
    if name == "beam":
        return BeamSearchSolver()
    if name == "multi_rule_greedy":
        return MultiRuleGreedySolver()
    if name == "lpfh_multi_heuristic":
        return LpfhMultiHeuristicSolver()
    if name == "target_guided_beam":
        return TargetGuidedBeamSolver()
    if name == "wang_bs_g":
        return WangBsGiantMovesSolver()
    if name == "wang_bs_b":
        return WangBsBabyMovesSolver()
    if name == "brkga_cpmp":
        return BrkgaCpmpSolver()
    if name == "bs_frg":
        return BsFrgSolver()
    raise ValueError(f"unknown classical CPMP solver: {name}")


def _priority_group_sorted_stack(stack: list[int]) -> bool:
    return all(stack[idx] >= stack[idx + 1] for idx in range(len(stack) - 1))


def _simulate_priority_group_solution(instance: Instance, solution: Solution) -> tuple[bool, str, list[list[int]], int | str]:
    stacks = [list(stack) for stack in instance.stacks]
    last_step = -1
    try:
        for move in solution.moves:
            if move.step < last_step:
                return False, "move steps are not nondecreasing", stacks, future_disorder_count_stacks(stacks)
            last_step = move.step
            if move.from_stack == move.to_stack:
                return False, "source and destination must differ", stacks, future_disorder_count_stacks(stacks)
            if not (0 <= move.from_stack < instance.stack_count):
                return False, "source stack out of range", stacks, future_disorder_count_stacks(stacks)
            if not (0 <= move.to_stack < instance.stack_count):
                return False, "destination stack out of range", stacks, future_disorder_count_stacks(stacks)
            src = stacks[move.from_stack]
            dst = stacks[move.to_stack]
            if not src:
                return False, "source stack is empty", stacks, future_disorder_count_stacks(stacks)
            if len(dst) >= instance.max_height:
                return False, "destination stack is full", stacks, future_disorder_count_stacks(stacks)
            if src[-1] != move.box:
                return False, f"move box {move.box} is not source top {src[-1]}", stacks, future_disorder_count_stacks(stacks)
            dst.append(src.pop())
    except Exception as exc:  # noqa: BLE001 - row writer should report validation status.
        return False, str(exc), stacks, ""
    final_n = future_disorder_count_stacks(stacks)
    if not all(_priority_group_sorted_stack(stack) for stack in stacks):
        return False, "final state is not a priority-group goal", stacks, final_n
    return True, "ok", stacks, final_n


def _final_n_after_moves(instance: Instance, solution: Solution, allow_priority_groups: bool = False) -> int | str:
    if allow_priority_groups:
        _ok, _msg, _stacks, final_n = _simulate_priority_group_solution(instance, solution)
        return final_n
    try:
        state = create_initial_state(instance)
        for move in solution.moves:
            state = apply_move(instance, state, move)
        return disorder_box_count(instance, state)
    except Exception:  # noqa: BLE001 - row writer should report validation status.
        return ""


def row_from_solution(
    instance: Instance,
    family: str,
    case_id: str,
    solution: Solution,
    reference_moves: int | None = None,
    allow_priority_groups: bool = False,
) -> dict[str, object]:
    if solution.moves or solution.feasible:
        if allow_priority_groups:
            goal_ok, validation_msg, _stacks, _final_n = _simulate_priority_group_solution(instance, solution)
        else:
            goal_ok, validation_msg, _stacks = validate_solution(instance, solution)
    else:
        goal_ok, validation_msg, _stacks = False, solution.status, []
    moves: int | str = len(solution.moves) if solution.feasible else ""
    final_n = _final_n_after_moves(instance, solution, allow_priority_groups=allow_priority_groups) if solution.moves or solution.feasible else ""
    return {
        "family": family,
        "case": case_id,
        "solver": solution.solver_name,
        "moves": moves,
        "final_n": final_n,
        "reference_moves": reference_moves if reference_moves is not None else "",
        "move_gap": (int(moves) - reference_moves) if moves != "" and reference_moves is not None else "",
        "runtime_sec": solution.runtime_sec,
        "expanded_states": solution.expanded_states,
        "generated_states": solution.generated_states,
        "visited_states": solution.visited_states,
        "goal_validation_ok": goal_ok,
        "validation_msg": validation_msg,
        "status": solution.status,
    }


def _unsupported_priority_group_row(instance: Instance, family: str, case_id: str, solver_name: str, reference_moves: int | None) -> dict[str, object]:
    solution = failure_solution(solver_name, time.perf_counter(), "UNSUPPORTED_PRIORITY_GROUPS")
    return row_from_solution(instance, family, case_id, solution, reference_moves)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "external" / "DLTS")
    parser.add_argument("--families", nargs="+", default=["5x5_cv1_test"])
    parser.add_argument("--limit-per-family", type=int, default=5)
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["beam", "multi_rule_greedy", "lpfh_multi_heuristic", "target_guided_beam", "brkga_cpmp"],
    )
    parser.add_argument("--time-limit", type=float, default=3.0)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--beam-width", type=int, default=256)
    parser.add_argument("--max-candidates-per-layer", type=int, default=4096)
    parser.add_argument("--entropy-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "experiments" / "classical_cpmp_baselines.csv")
    return parser


def main() -> int:
    from open_cpmp_benchmarks import discover_dlts_cases, load_dlts_dat

    parser = build_parser()
    args = parser.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    cases = discover_dlts_cases(source, families=args.families, limit_per_family=args.limit_per_family)
    rows: list[dict[str, object]] = []

    for case in cases:
        instance = load_dlts_dat(case.instance_path, max_height=case.reference.max_height)
        reference_moves = case.reference.optimal_moves
        try:
            validate_instance(instance)
            priority_groups_supported = True
        except ValueError:
            priority_groups_supported = False

        for solver_name in args.solvers:
            allow_priority_groups = (not priority_groups_supported) and solver_name in PRIORITY_GROUP_SOLVERS
            if not priority_groups_supported and not allow_priority_groups:
                rows.append(_unsupported_priority_group_row(instance, case.family, case.case_id, solver_name, reference_moves))
                continue
            solver = build_solver(solver_name)
            config = SolverConfig(
                max_depth=args.max_depth,
                beam_width=args.beam_width,
                max_candidates_per_layer=args.max_candidates_per_layer,
                time_limit_sec=args.time_limit,
                entropy_alpha=args.entropy_alpha,
                random_seed=args.seed,
            )
            print(f"CASE_START {case.case_id} {solver_name}", flush=True)
            solution = solver.solve(instance, config)
            row = row_from_solution(instance, case.family, case.case_id, solution, reference_moves, allow_priority_groups=allow_priority_groups)
            rows.append(row)
            print(
                f"CASE_DONE {case.case_id} {solver_name} final_n={row['final_n']} moves={row['moves']} "
                f"runtime={float(row['runtime_sec']):.3f}s status={row['status']}",
                flush=True,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(output, rows)
    print(f"WROTE {output}", flush=True)
    return 0


def _write_summary(output: Path, rows: list[dict[str, object]]) -> None:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["solver"]), []).append(row)
    summary_path = output.with_name(output.stem + "_summary.md")
    lines = ["# Classical CPMP Baseline Summary", "", f"Result CSV: `{output}`", "", "| solver | solved | moves_sum | avg_runtime_sec |", "|---|---:|---:|---:|"]
    for solver, solver_rows in sorted(groups.items()):
        solved_rows = [row for row in solver_rows if row["final_n"] == 0 and row["moves"] != ""]
        moves_sum = sum(int(row["moves"]) for row in solved_rows)
        avg_runtime = sum(float(row["runtime_sec"]) for row in solver_rows) / max(1, len(solver_rows))
        lines.append(f"| {solver} | {len(solved_rows)}/{len(solver_rows)} | {moves_sum if solved_rows else ''} | {avg_runtime:.3f} |")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
