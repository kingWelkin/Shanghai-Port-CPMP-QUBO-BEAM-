"""Exact A* and time-aligned traditional wide-beam baselines.

This benchmark is intended as a control group for newer QUBO / macro / hybrid
routes.  For each single-bay real case it runs:

* exact_astar: strict A* over the full legal relocation graph.
* traditional_wide_beam_time_aligned: plain classical beam with no QUBO and no
  macro actions.  The script sweeps wider beam widths and selects the run whose
  runtime is closest to a target runtime, optionally read from a reference CSV.

The beam sweep deliberately uses the same scoring/legality layer as the older
business beam baseline; increasing width is the only intended knob.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_explainable_rule_macro_beam import _write_explain_csv  # noqa: E402
from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from benchmark_full_real_business_algorithms import (  # noqa: E402
    TrajectoryResult,
    _result_from_state,
)
from container_reshuffle.core import (  # noqa: E402
    Instance,
    Move,
    Solution,
    State,
    apply_move,
    create_initial_state,
    generate_legal_moves,
    hash_state,
    validate_solution,
)
from container_reshuffle.features import (  # noqa: E402
    disorder_box_count,
    entropy_limit,
    entropy_value,
    max_entropy_along_moves,
)
from container_reshuffle.scoring import cpmp_business_feasible, dominance_value, score_state  # noqa: E402
from container_reshuffle.solvers.astar import ExactAStarSolver  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from container_reshuffle.solvers.beam import Candidate  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


RESULT_FIELDS = list(TrajectoryResult.__annotations__.keys())
ACCEPTANCE_FIELDS = [
    "case",
    "solver",
    "selected",
    "beam_width",
    "target_runtime_sec",
    "runtime_gap_sec",
    "final_n",
    "moves",
    "runtime_sec",
    "business_ok",
    "goal_validation_ok",
    "validation_msg",
    "moves_path",
    "status",
]
SWEEP_EXTRA_FIELDS = [
    "beam_width",
    "target_runtime_sec",
    "runtime_gap_sec",
    "selected",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _stack_label_to_index(label: str) -> int:
    value = str(label).strip().upper()
    if value == "A":
        return 9
    return int(value) - 1


def _read_explain_moves(path: Path) -> list[Move]:
    rows = _read_rows(path)
    moves: list[Move] = []
    for row in rows:
        if not row.get("box_rank"):
            continue
        step = int(float(row.get("step") or len(moves) + 1)) - 1
        moves.append(
            Move(
                step=step,
                box=int(float(row["box_rank"])),
                from_stack=_stack_label_to_index(row["from_stack"]),
                to_stack=_stack_label_to_index(row["to_stack"]),
            )
        )
    return moves


def _incumbent_moves_by_case(args: argparse.Namespace, cases: Iterable[str]) -> dict[str, list[Move]]:
    if args.incumbent_acceptance is None:
        return {}
    path = args.incumbent_acceptance if args.incumbent_acceptance.is_absolute() else ROOT / args.incumbent_acceptance
    out: dict[str, list[Move]] = {}
    wanted = set(cases)
    for row in _read_rows(path):
        case = row.get("case", "")
        if case not in wanted:
            continue
        if row.get("goal_validation_ok") != "True" or row.get("final_n") not in {"0", "0.0"}:
            continue
        move_path_text = row.get("moves_path", "")
        if not move_path_text:
            continue
        move_path = Path(move_path_text)
        if not move_path.is_absolute():
            move_path = ROOT / move_path
        if not move_path.exists():
            continue
        out[case] = _read_explain_moves(move_path)
    return out


def _parse_case(case: str) -> tuple[str, str]:
    parts = case.split("_")
    if len(parts) < 2:
        raise ValueError(f"cannot parse case name: {case}")
    return parts[0], parts[1]


def _normalize_case(case: str) -> str:
    return case if case.endswith("_single") else f"{case}_single"


def _case_filter(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return [_normalize_case(case) for case in args.cases]
    return [
        f"{zone}_{bay}_single"
        for zone, bay in discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))
    ]


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _state_from_moves(instance: Instance, moves) -> State:
    state = create_initial_state(instance)
    for move in moves:
        state = apply_move(instance, state, move)
    return state


def _failure_row(
    case: str,
    solver: str,
    instance: Instance,
    runtime_sec: float,
    status: str,
    *,
    expanded_states: int = 0,
    generated_states: int = 0,
    visited_states: int = 0,
) -> TrajectoryResult:
    initial = create_initial_state(instance)
    n0 = disorder_box_count(instance, initial)
    cap13 = entropy_limit(instance, 1.3)
    cap15 = entropy_limit(instance, 1.5)
    return TrajectoryResult(
        case=case,
        solver=solver,
        n0=n0,
        cap_13=cap13,
        cap_15=cap15,
        moves=None,
        final_n=None,
        reduction=None,
        g_end=None,
        max_g=None,
        g_ok_13=None,
        g_ok_15=None,
        runtime_sec=runtime_sec,
        expanded_states=expanded_states,
        generated_states=generated_states,
        visited_states=visited_states,
        status=status,
    )


def _row_from_solution(case: str, solver_name: str, instance: Instance, solution: Solution, start: float) -> tuple[TrajectoryResult, State | None]:
    if not solution.feasible:
        return (
            _failure_row(
                case,
                solver_name,
                instance,
                solution.runtime_sec,
                solution.status,
                expanded_states=solution.expanded_states,
                generated_states=solution.generated_states,
                visited_states=solution.visited_states,
            ),
            None,
        )

    state = _state_from_moves(instance, solution.moves)
    row = _result_from_state(
        case,
        solver_name,
        instance,
        state,
        start,
        expanded_states=solution.expanded_states,
        generated_states=solution.generated_states,
        visited_states=solution.visited_states,
        status=solution.status + f"; optimal={solution.optimal}",
    )
    row.runtime_sec = solution.runtime_sec
    return row, state


def _target_runtime_by_case(args: argparse.Namespace, cases: Iterable[str]) -> dict[str, float]:
    default = float(args.target_runtime_sec)
    out = {case: default for case in cases}
    if args.reference_results is None:
        return out
    ref_path = args.reference_results if args.reference_results.is_absolute() else ROOT / args.reference_results
    rows = _read_rows(ref_path)
    if not rows:
        return out
    for case in out:
        matches = [row for row in rows if row.get("case") == case]
        if args.reference_solver:
            solver_matches = [row for row in matches if row.get("solver") == args.reference_solver]
            if solver_matches:
                matches = solver_matches
        if not matches:
            continue
        best = min(
            matches,
            key=lambda row: (
                int(float(row.get("final_n") or 10**9)),
                int(float(row.get("moves") or 10**9)),
                float(row.get("runtime_sec") or default),
            ),
        )
        try:
            out[case] = max(args.min_target_runtime_sec, float(best.get("runtime_sec") or default))
        except ValueError:
            out[case] = default
    return out


def _beam_config(args: argparse.Namespace, width: int, target_runtime: float) -> SolverConfig:
    multiplier = max(1.0, float(args.max_candidates_multiplier))
    max_candidates = int(max(width, round(width * multiplier)))
    time_limit = max(args.min_beam_time_limit_sec, target_runtime * args.beam_time_limit_factor)
    return SolverConfig(
        max_depth=args.max_depth,
        beam_width=width,
        max_candidates_per_layer=max_candidates,
        candidate_limit_per_state=args.candidate_limit_per_state,
        time_limit_sec=time_limit,
        entropy_alpha=args.alpha,
        random_seed=args.seed,
        enable_canonical_hash=args.canonical,
    )


def _beam_selection_key(row: TrajectoryResult, width: int, target_runtime: float) -> tuple:
    final_n = row.final_n if row.final_n is not None else 10**9
    moves = row.moves if row.moves is not None else 10**9
    runtime_gap = abs(row.runtime_sec - target_runtime)
    return (
        runtime_gap,
        final_n,
        moves,
        -width,
    )


def _state_rank(instance: Instance, state: State) -> tuple[int, int, int, int]:
    return (
        disorder_box_count(instance, state),
        state.disorder_total,
        state.entropy_max,
        state.move_count,
    )


def _run_traditional_beam(
    case: str,
    instance: Instance,
    config: SolverConfig,
    *,
    solver_name: str,
) -> tuple[TrajectoryResult, State]:
    """Run plain beam and return the best state, including its moves."""

    start = time.perf_counter()
    initial = create_initial_state(instance)
    best = initial
    h0 = hash_state(instance, initial, config.enable_canonical_hash)
    beam = [Candidate(initial, score_state(instance, initial, config.score_weights), None, h0)]
    visited = {h0: dominance_value(instance, initial)}
    expanded = generated = 0

    for _depth in range(config.max_depth):
        if time.perf_counter() - start > config.time_limit_sec:
            break
        candidates: list[Candidate] = []
        for cand in beam:
            if time.perf_counter() - start > config.time_limit_sec:
                break
            expanded += 1
            scored_moves = []
            for move in generate_legal_moves(instance, cand.state):
                try:
                    nxt = apply_move(instance, cand.state, move)
                except ValueError:
                    continue
                if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                    continue
                scored_moves.append((score_state(instance, nxt, config.score_weights), move, nxt))
            scored_moves.sort(key=lambda item: item[0])
            if config.candidate_limit_per_state is not None:
                scored_moves = scored_moves[: config.candidate_limit_per_state]
            for score, move, nxt in scored_moves:
                h = hash_state(instance, nxt, config.enable_canonical_hash)
                dom = dominance_value(instance, nxt)
                if config.enable_visited_pruning and visited.get(h, float("inf")) <= dom:
                    continue
                visited[h] = dom
                if _state_rank(instance, nxt) < _state_rank(instance, best):
                    best = nxt
                candidates.append(Candidate(nxt, score, move, h))
        generated += len(candidates)
        if not candidates or disorder_box_count(instance, best) == 0:
            break
        candidates.sort(key=lambda cand: (cand.score, cand.state.move_count))
        if config.max_candidates_per_layer:
            candidates = candidates[: config.max_candidates_per_layer]
        beam = candidates[: config.beam_width]

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    row = _result_from_state(
        case,
        solver_name,
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=len(visited),
        status=status,
    )
    return row, best


def _run_wide_beam_sweep(
    case: str,
    instance: Instance,
    args: argparse.Namespace,
    target_runtime: float,
) -> tuple[TrajectoryResult, State, list[TrajectoryResult], dict[str, int]]:
    sweep_rows: list[TrajectoryResult] = []
    sweep_states: dict[int, State] = {}
    selected: TrajectoryResult | None = None
    selected_width = 0
    widths = sorted(set(int(width) for width in args.beam_widths))
    for width in widths:
        config = _beam_config(args, width, target_runtime)
        row, state = _run_traditional_beam(
            case,
            instance,
            config,
            solver_name=f"traditional_beam_w{width}",
        )
        row.status += (
            f"; beam_width={width}; max_candidates_per_layer={config.max_candidates_per_layer}; "
            f"target_runtime_sec={target_runtime:.6f}"
        )
        sweep_rows.append(row)
        sweep_states[width] = state
        if selected is None or _beam_selection_key(row, width, target_runtime) < _beam_selection_key(selected, selected_width, target_runtime):
            selected = row
            selected_width = width
        print(
            "  BEAM_SWEEP "
            f"case={case} width={width} finalN={row.final_n} moves={row.moves} "
            f"runtime={row.runtime_sec:.3f}s target={target_runtime:.3f}s",
            flush=True,
        )
        if args.stop_after_runtime_match and row.runtime_sec >= target_runtime:
            break
    if selected is None:
        raise RuntimeError("beam sweep produced no rows")
    selected_row = TrajectoryResult(**selected.__dict__)
    selected_row.solver = "traditional_wide_beam_time_aligned"
    selected_row.status += f"; selected_from_width={selected_width}"
    return selected_row, sweep_states[selected_width], sweep_rows, {"selected_width": selected_width}


def _validate_row(instance: Instance, row: TrajectoryResult, state: State | None) -> tuple[bool, bool, str]:
    if state is None or row.moves is None:
        return False, False, row.status
    try:
        max_g = max_entropy_along_moves(instance, state.moves) if state.moves else entropy_value(instance, state)
    except Exception as exc:  # noqa: BLE001
        return False, False, str(exc)
    business_ok = max_g <= entropy_limit(instance, 1.3)
    if row.final_n != 0:
        return business_ok, False, "best-effort state is not a terminal CPMP goal"
    sol = Solution(
        solver_name=row.solver,
        feasible=True,
        optimal=False,
        objective=float(len(state.moves)),
        runtime_sec=row.runtime_sec,
        moves=state.moves,
        status=row.status,
    )
    goal_ok, goal_msg, _ = validate_solution(instance, sol)
    return business_ok, goal_ok, goal_msg


def _write_result_rows(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _write_sweep_rows(rows: list[dict], path: Path) -> None:
    fields = RESULT_FIELDS + SWEEP_EXTRA_FIELDS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_acceptance(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ACCEPTANCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(result_rows: list[TrajectoryResult], sweep_path: Path, acceptance_path: Path, output_path: Path, report_path: Path) -> None:
    by_solver: dict[str, list[TrajectoryResult]] = {}
    for row in result_rows:
        by_solver.setdefault(row.solver, []).append(row)
    lines = [
        "# Exact A* And Time-Aligned Wide Beam Baselines",
        "",
        f"Result CSV: `{_rel(output_path)}`",
        f"Beam sweep CSV: `{_rel(sweep_path)}`",
        f"Acceptance CSV: `{_rel(acceptance_path)}`",
        "",
        "| solver | cases | rows with final N | N=0 solved | avg known final N | avg known moves | avg runtime sec |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for solver, rows in sorted(by_solver.items()):
        solved = [row for row in rows if row.final_n == 0]
        finals = [row.final_n for row in rows if row.final_n is not None]
        moves = [row.moves for row in rows if row.moves is not None]
        avg_final = sum(finals) / len(finals) if finals else 0.0
        avg_moves = sum(moves) / len(moves) if moves else 0.0
        avg_runtime = sum(row.runtime_sec for row in rows) / len(rows)
        lines.append(
            f"| `{solver}` | {len(rows)} | {len(finals)}/{len(rows)} | {len(solved)}/{len(rows)} | "
            f"{avg_final:.3f} | {avg_moves:.3f} | {avg_runtime:.3f} |"
        )
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["exact_astar", "wide_beam"],
        default=["exact_astar", "wide_beam"],
    )
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "experiments" / "exact_astar_wide_beam_baselines.csv")
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--reference-solver", default="")
    parser.add_argument("--incumbent-acceptance", type=Path)
    parser.add_argument("--target-runtime-sec", type=float, default=8.0)
    parser.add_argument("--min-target-runtime-sec", type=float, default=0.1)
    parser.add_argument("--exact-time-limit", type=float, default=30.0)
    parser.add_argument("--exact-max-visited-states", type=int, default=250_000)
    parser.add_argument("--beam-time-limit-factor", type=float, default=1.25)
    parser.add_argument("--min-beam-time-limit-sec", type=float, default=1.0)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--beam-widths", nargs="+", type=int, default=[10, 20, 40, 80, 160, 320, 640, 1200])
    parser.add_argument("--max-candidates-multiplier", type=float, default=8.0)
    parser.add_argument("--candidate-limit-per-state", type=int)
    parser.add_argument("--stop-after-runtime-match", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--canonical", action="store_true")
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    sweep_output = output.with_name(output.stem + "_beam_sweep.csv")
    acceptance_output = output.with_name(output.stem + "_acceptance.csv")
    report_output = output.with_suffix(".md")
    moves_dir = output.with_name(output.stem + "_moves")
    moves_dir.mkdir(parents=True, exist_ok=True)

    cases = _case_filter(args)
    target_runtimes = _target_runtime_by_case(args, cases)
    incumbent_moves = _incumbent_moves_by_case(args, cases)
    result_rows: list[TrajectoryResult] = []
    sweep_dicts: list[dict] = []
    acceptance_rows: list[dict] = []

    for index, case in enumerate(cases, start=1):
        zone, bay = _parse_case(case)
        instance = _instance_from_single_bay(zone, bay)
        target_runtime = target_runtimes[case]
        print(
            "CASE_START "
            f"{index}/{len(cases)} case={case} N0={disorder_box_count(instance, create_initial_state(instance))} "
            f"cap={entropy_limit(instance, args.alpha)} target_runtime={target_runtime:.3f}s",
            flush=True,
        )

        states_by_solver: dict[str, State | None] = {}
        meta_by_solver: dict[str, dict] = {}

        if "exact_astar" in args.methods:
            start = time.perf_counter()
            exact_config = SolverConfig(
                max_depth=min(args.max_depth, entropy_limit(instance, args.alpha)),
                time_limit_sec=args.exact_time_limit,
                exact_max_visited_states=args.exact_max_visited_states,
                exact_incumbent_moves=incumbent_moves.get(case),
                entropy_alpha=args.alpha,
                enable_canonical_hash=args.canonical,
            )
            solution = ExactAStarSolver().solve(instance, exact_config)
            exact_row, exact_state = _row_from_solution(case, "exact_astar", instance, solution, start)
            result_rows.append(exact_row)
            states_by_solver[exact_row.solver] = exact_state
            meta_by_solver[exact_row.solver] = {
                "beam_width": "",
                "target_runtime_sec": target_runtime,
                "runtime_gap_sec": "",
            }
            print(
                "  EXACT_ASTAR "
                f"status={exact_row.status} finalN={exact_row.final_n} moves={exact_row.moves} "
                f"runtime={exact_row.runtime_sec:.3f}s warm_start_moves={len(incumbent_moves.get(case, []))}",
                flush=True,
            )

        if "wide_beam" in args.methods:
            selected_row, selected_state, sweep_rows, selected_meta = _run_wide_beam_sweep(case, instance, args, target_runtime)
            result_rows.append(selected_row)
            states_by_solver[selected_row.solver] = selected_state
            meta_by_solver[selected_row.solver] = {
                "beam_width": selected_meta["selected_width"],
                "target_runtime_sec": target_runtime,
                "runtime_gap_sec": abs(selected_row.runtime_sec - target_runtime),
            }
            for row in sweep_rows:
                width = int(row.solver.rsplit("w", 1)[1])
                sweep_dict = row.__dict__.copy()
                sweep_dict.update(
                    {
                        "beam_width": width,
                        "target_runtime_sec": target_runtime,
                        "runtime_gap_sec": abs(row.runtime_sec - target_runtime),
                        "selected": width == selected_meta["selected_width"],
                    }
                )
                sweep_dicts.append(sweep_dict)
            print(
                "  WIDE_BEAM_SELECTED "
                f"width={selected_meta['selected_width']} finalN={selected_row.final_n} "
                f"moves={selected_row.moves} runtime={selected_row.runtime_sec:.3f}s",
                flush=True,
            )

        for row in result_rows:
            if row.case != case:
                continue
            state = states_by_solver.get(row.solver)
            business_ok, goal_ok, msg = _validate_row(instance, row, state)
            move_path = Path("")
            if state is not None and row.moves:
                move_path = moves_dir / f"{row.case}_{row.solver}_moves.csv"
                _write_explain_csv(instance, state.moves, move_path)
            meta = meta_by_solver.get(row.solver, {})
            acceptance_rows.append(
                {
                    "case": row.case,
                    "solver": row.solver,
                    "selected": "True",
                    "beam_width": meta.get("beam_width", ""),
                    "target_runtime_sec": meta.get("target_runtime_sec", ""),
                    "runtime_gap_sec": meta.get("runtime_gap_sec", ""),
                    "final_n": row.final_n,
                    "moves": row.moves,
                    "runtime_sec": f"{row.runtime_sec:.6f}",
                    "business_ok": str(business_ok),
                    "goal_validation_ok": str(goal_ok),
                    "validation_msg": msg,
                    "moves_path": _rel(move_path) if move_path else "",
                    "status": row.status,
                }
            )

        _write_result_rows(result_rows, output)
        _write_sweep_rows(sweep_dicts, sweep_output)
        _write_acceptance(acceptance_rows, acceptance_output)
        _write_report(result_rows, sweep_output, acceptance_output, output, report_output)

    print(f"wrote {output}", flush=True)
    print(f"wrote {sweep_output}", flush=True)
    print(f"wrote {acceptance_output}", flush=True)
    print(f"wrote {report_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
