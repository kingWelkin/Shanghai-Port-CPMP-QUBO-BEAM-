"""Tune a protected fast-slack + QUBO portfolio on real single-bay cases."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_explainable_rule_macro_beam import _write_explain_csv  # noqa: E402
from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from benchmark_fast_slack_beam import run_fast_slack_beam  # noqa: E402
from benchmark_fast_slack_qubo_beam import run_fast_slack_qubo_beam  # noqa: E402
from benchmark_full_real_business_algorithms import TrajectoryResult  # noqa: E402
from container_reshuffle.core import (  # noqa: E402
    Instance,
    Move,
    Solution,
    State,
    apply_move,
    create_initial_state,
    validate_solution,
)
from container_reshuffle.features import disorder_box_count, entropy_limit  # noqa: E402
from container_reshuffle.scoring import cpmp_business_feasible  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


HARD_CASES = [
    "A1_26_single",
    "A2_26_single",
    "A3_26_single",
    "A5_26_single",
    "B1_26_single",
    "B2_26_single",
    "B3_26_single",
    "B4_26_single",
    "B5_26_single",
    "C2_26_single",
    "C4_26_single",
    "C5_26_single",
]

SEED_SPECS = [
    (
        "fast_slack_8s_w120",
        ROOT / "docs" / "experiments" / "fast_slack_beam_all_single_bays_8s_w120.csv",
        ROOT / "docs" / "experiments" / "fast_slack_beam_all_single_moves_8s_w120",
        False,
        "seed_fast",
    ),
    (
        "fast_slack_qubo_8s_plateau",
        ROOT / "docs" / "experiments" / "fast_slack_qubo_beam_all_single_bays_8s_plateau.csv",
        ROOT / "docs" / "experiments" / "fast_slack_qubo_beam_all_single_moves_8s_plateau",
        True,
        "seed_qubo",
    ),
    (
        "fast_slack_rescue18_w240",
        ROOT / "docs" / "experiments" / "fast_slack_rescue_all_single_bays_18s_w240.csv",
        ROOT / "docs" / "experiments" / "fast_slack_rescue_all_single_moves_18s_w240",
        False,
        "seed_rescue",
    ),
]

EXACT_REFERENCE = ROOT / "docs" / "experiments" / "fast_slack_vs_exact_all_single_bays_8s_w120.csv"

ALL_CANDIDATE_FIELDS = [
    "case",
    "family",
    "variant",
    "solver",
    "final_n",
    "moves",
    "runtime_sec",
    "exact_moves",
    "move_gap",
    "qubo_used",
    "qubo_calls",
    "qubo_total_vars",
    "qubo_time_sec",
    "business_ok",
    "goal_validation_ok",
    "validation_msg",
    "status",
    "source",
    "moves_path",
]

PORTFOLIO_FIELDS = [
    "case",
    "selected_solver",
    "final_n",
    "moves",
    "runtime_sec",
    "exact_moves",
    "move_gap",
    "qubo_used",
    "selected_reason",
    "business_ok",
    "goal_validation_ok",
    "validation_msg",
    "moves_path",
]


@dataclass
class CandidateRecord:
    """One concrete candidate trajectory plus enough metadata to select it."""

    case: str
    family: str
    variant: str
    result: TrajectoryResult
    moves: list[Move]
    qubo_used: bool
    source: str
    moves_path: Path | None = None
    business_ok: bool = False
    goal_validation_ok: bool = False
    validation_msg: str = ""


def _parse_case(case: str) -> tuple[str, str]:
    parts = case.split("_")
    if len(parts) < 2:
        raise ValueError(f"cannot parse case name: {case}")
    return parts[0], parts[1]


def _label_to_index(label: str) -> int:
    label = str(label).strip()
    if label.upper() == "A":
        return 9
    return int(label) - 1


def _read_moves(path: Path) -> list[Move]:
    if not path.exists():
        return []
    moves: list[Move] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("step"):
                continue
            moves.append(
                Move(
                    step=int(row["step"]) - 1,
                    box=int(row["box_rank"]),
                    from_stack=_label_to_index(row["from_stack"]),
                    to_stack=_label_to_index(row["to_stack"]),
                )
            )
    return moves


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _exact_moves_by_case(path: Path = EXACT_REFERENCE) -> dict[str, int]:
    refs: dict[str, int] = {}
    for row in _read_rows(path):
        value = row.get("exact_moves_to_n0") or row.get("exact_moves")
        if value not in (None, ""):
            refs[row["case"]] = int(float(value))
    return refs


def _row_int(row: dict[str, str], key: str) -> int | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return int(float(value))


def _row_float(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    return 0.0 if value in (None, "") else float(value)


def _result_from_row(row: dict[str, str], solver_name: str | None = None) -> TrajectoryResult:
    return TrajectoryResult(
        case=row["case"],
        solver=solver_name or row.get("solver", ""),
        n0=_row_int(row, "n0"),
        cap_13=_row_int(row, "cap_13"),
        cap_15=_row_int(row, "cap_15"),
        moves=_row_int(row, "moves"),
        final_n=_row_int(row, "final_n"),
        reduction=_row_int(row, "reduction"),
        g_end=_row_int(row, "g_end"),
        max_g=_row_int(row, "max_g"),
        g_ok_13=str(row.get("g_ok_13", "")).lower() == "true",
        g_ok_15=str(row.get("g_ok_15", "")).lower() == "true",
        runtime_sec=_row_float(row, "runtime_sec"),
        expanded_states=_row_int(row, "expanded_states") or 0,
        generated_states=_row_int(row, "generated_states") or 0,
        visited_states=_row_int(row, "visited_states") or 0,
        qubo_calls=_row_int(row, "qubo_calls") or 0,
        qubo_total_vars=_row_int(row, "qubo_total_vars") or 0,
        qubo_time_sec=_row_float(row, "qubo_time_sec"),
        status=row.get("status", ""),
    )


def _simulate_business(instance: Instance, moves: list[Move]) -> tuple[bool, str, State]:
    state = create_initial_state(instance)
    cap = entropy_limit(instance, 1.3)
    try:
        for move in moves:
            state = apply_move(instance, state, move)
            if state.entropy_max > cap:
                return False, f"G_LIMIT_EXCEEDED max_g={state.entropy_max} cap={cap}", state
            if not cpmp_business_feasible(instance, state, 1.3):
                return False, "BUSINESS_INFEASIBLE", state
        return True, "ok", state
    except Exception as exc:  # noqa: BLE001 - diagnostics should not abort tuning.
        return False, str(exc), state


def _validate_candidate(candidate: CandidateRecord, instance: Instance) -> CandidateRecord:
    business_ok, business_msg, state = _simulate_business(instance, candidate.moves)
    final_n = disorder_box_count(instance, state)
    if candidate.result.final_n is not None and final_n != candidate.result.final_n:
        business_ok = False
        business_msg = f"{business_msg}; FINAL_N_MISMATCH simulated={final_n} row={candidate.result.final_n}"
    goal_ok = False
    goal_msg = "not N_ZERO"
    if final_n == 0:
        solution = Solution(
            solver_name=candidate.result.solver,
            feasible=True,
            optimal=False,
            objective=float(len(candidate.moves)),
            runtime_sec=candidate.result.runtime_sec,
            moves=candidate.moves,
            status=candidate.result.status,
        )
        goal_ok, goal_msg, _ = validate_solution(instance, solution)
    candidate.business_ok = business_ok
    candidate.goal_validation_ok = goal_ok
    candidate.validation_msg = goal_msg if final_n == 0 else business_msg
    if final_n != 0 and business_ok:
        candidate.validation_msg = business_msg
    return candidate


def _display_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _candidate_from_result(
    *,
    family: str,
    variant: str,
    result: TrajectoryResult,
    state: State,
    qubo_used: bool,
    source: str,
    moves_path: Path | None = None,
) -> CandidateRecord:
    return CandidateRecord(
        case=result.case,
        family=family,
        variant=variant,
        result=result,
        moves=list(state.moves),
        qubo_used=qubo_used,
        source=source,
        moves_path=moves_path,
    )


def _load_seed_candidates(exact_refs: dict[str, int]) -> list[CandidateRecord]:
    candidates: list[CandidateRecord] = []
    for solver_name, csv_path, moves_dir, qubo_used, family in SEED_SPECS:
        for row in _read_rows(csv_path):
            case = row["case"]
            move_path = moves_dir / f"{case}_moves.csv"
            moves = _read_moves(move_path)
            if not moves and int(float(row.get("moves", "0") or 0)) > 0:
                continue
            result = _result_from_row(row, solver_name=solver_name)
            result.qubo_calls = max(result.qubo_calls, 1 if qubo_used else 0)
            if qubo_used and result.qubo_total_vars == 0:
                result.qubo_total_vars = 1
            candidate = CandidateRecord(
                case=case,
                family=family,
                variant=solver_name,
                result=result,
                moves=moves,
                qubo_used=qubo_used,
                source=str(csv_path.relative_to(ROOT)),
                moves_path=move_path,
            )
            zone, bay = _parse_case(case)
            candidate = _validate_candidate(candidate, _instance_from_single_bay(zone, bay))
            candidates.append(candidate)
    return candidates


def _result_key(candidate: CandidateRecord) -> tuple[int, int, float, int]:
    final_n = candidate.result.final_n if candidate.result.final_n is not None else 10**9
    moves = candidate.result.moves if candidate.result.moves is not None else 10**9
    return final_n, moves, candidate.result.runtime_sec, 0 if candidate.qubo_used else 1


def _best_rule_candidate(candidates: list[CandidateRecord]) -> CandidateRecord | None:
    rules = [candidate for candidate in candidates if not candidate.qubo_used]
    return min(rules, key=_result_key) if rules else None


def _eligible_for_portfolio(candidate: CandidateRecord, case_candidates: list[CandidateRecord]) -> bool:
    if not candidate.business_ok:
        return False
    if not candidate.qubo_used:
        return True
    best_rule = _best_rule_candidate(case_candidates)
    if best_rule is None:
        return True
    cand_final = candidate.result.final_n if candidate.result.final_n is not None else 10**9
    rule_final = best_rule.result.final_n if best_rule.result.final_n is not None else 10**9
    cand_moves = candidate.result.moves if candidate.result.moves is not None else 10**9
    rule_moves = best_rule.result.moves if best_rule.result.moves is not None else 10**9
    return cand_final <= rule_final and cand_moves <= rule_moves


def _select_portfolio(candidates: list[CandidateRecord], *, include_rescue: bool) -> dict[str, CandidateRecord]:
    selected: dict[str, CandidateRecord] = {}
    cases = sorted({candidate.case for candidate in candidates})
    for case in cases:
        rows = [candidate for candidate in candidates if candidate.case == case]
        usable = [
            candidate
            for candidate in rows
            if _eligible_for_portfolio(candidate, rows)
            and (include_rescue or candidate.family not in {"rescue_grid", "seed_rescue"})
        ]
        if not usable:
            usable = [candidate for candidate in rows if candidate.business_ok]
        if usable:
            selected[case] = min(usable, key=_result_key)
    return selected


def _move_gap(candidate: CandidateRecord, exact_refs: dict[str, int]) -> str:
    exact = exact_refs.get(candidate.case)
    if exact is None or candidate.result.final_n != 0 or candidate.result.moves is None:
        return ""
    return str(candidate.result.moves - exact)


def _candidate_row(candidate: CandidateRecord, exact_refs: dict[str, int]) -> dict[str, object]:
    exact = exact_refs.get(candidate.case, "")
    return {
        "case": candidate.case,
        "family": candidate.family,
        "variant": candidate.variant,
        "solver": candidate.result.solver,
        "final_n": candidate.result.final_n,
        "moves": candidate.result.moves,
        "runtime_sec": f"{candidate.result.runtime_sec:.6f}",
        "exact_moves": exact,
        "move_gap": _move_gap(candidate, exact_refs),
        "qubo_used": str(candidate.qubo_used),
        "qubo_calls": candidate.result.qubo_calls,
        "qubo_total_vars": candidate.result.qubo_total_vars,
        "qubo_time_sec": f"{candidate.result.qubo_time_sec:.6f}",
        "business_ok": str(candidate.business_ok),
        "goal_validation_ok": str(candidate.goal_validation_ok),
        "validation_msg": candidate.validation_msg,
        "status": candidate.result.status,
        "source": candidate.source,
        "moves_path": _display_path(candidate.moves_path),
    }


def _write_candidates(candidates: list[CandidateRecord], path: Path, exact_refs: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_CANDIDATE_FIELDS)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(_candidate_row(candidate, exact_refs))


def _selected_reason(candidate: CandidateRecord, case_candidates: list[CandidateRecord]) -> str:
    best_rule = _best_rule_candidate(case_candidates)
    if candidate.qubo_used:
        if best_rule is None:
            return "QUBO selected; no rule candidate"
        return (
            "QUBO selected without degrading rule core "
            f"(rule final_n={best_rule.result.final_n}, moves={best_rule.result.moves})"
        )
    if candidate.family in {"rescue_grid", "seed_rescue"}:
        return "rescue selected by final_n/moves/runtime"
    return "rule core selected by final_n/moves/runtime"


def _copy_or_write_moves(candidate: CandidateRecord, instance: Instance, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if candidate.moves_path is not None and candidate.moves_path.exists():
        shutil.copyfile(candidate.moves_path, path)
    else:
        _write_explain_csv(instance, candidate.moves, path)
    return path


def _write_portfolio(
    selected: dict[str, CandidateRecord],
    all_candidates: list[CandidateRecord],
    path: Path,
    moves_dir: Path,
    exact_refs: dict[str, int],
) -> None:
    rows: list[dict[str, object]] = []
    by_case = {}
    for candidate in all_candidates:
        by_case.setdefault(candidate.case, []).append(candidate)
    for case in sorted(selected):
        candidate = selected[case]
        zone, bay = _parse_case(case)
        move_path = _copy_or_write_moves(candidate, _instance_from_single_bay(zone, bay), moves_dir / f"{case}_moves.csv")
        rows.append(
            {
                "case": case,
                "selected_solver": candidate.variant,
                "final_n": candidate.result.final_n,
                "moves": candidate.result.moves,
                "runtime_sec": f"{candidate.result.runtime_sec:.6f}",
                "exact_moves": exact_refs.get(case, ""),
                "move_gap": _move_gap(candidate, exact_refs),
                "qubo_used": str(candidate.qubo_used),
                "selected_reason": _selected_reason(candidate, by_case.get(case, [])),
                "business_ok": str(candidate.business_ok),
                "goal_validation_ok": str(candidate.goal_validation_ok),
                "validation_msg": candidate.validation_msg,
                "moves_path": _display_path(move_path),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PORTFOLIO_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(output_prefix: Path, light: dict[str, CandidateRecord], quality: dict[str, CandidateRecord]) -> None:
    def summary(rows: dict[str, CandidateRecord]) -> tuple[int, float, float, int]:
        selected = list(rows.values())
        solved = sum(1 for row in selected if row.result.final_n == 0)
        avg_runtime = sum(row.result.runtime_sec for row in selected) / len(selected) if selected else 0.0
        gaps = []
        refs = _exact_moves_by_case()
        for row in selected:
            if row.result.final_n == 0 and row.result.moves is not None and row.case in refs:
                gaps.append(row.result.moves - refs[row.case])
        avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
        qubo_selected = sum(1 for row in selected if row.qubo_used)
        return solved, avg_runtime, avg_gap, qubo_selected

    light_solved, light_runtime, light_gap, light_qubo = summary(light)
    qual_solved, qual_runtime, qual_gap, qual_qubo = summary(quality)
    path = output_prefix.with_name(output_prefix.name + "_summary.md")
    lines = [
        "# Fast QUBO Portfolio Tuning",
        "",
        f"Candidate CSV: `{output_prefix.name}_all_candidates.csv`",
        f"Light CSV: `{output_prefix.name}_light.csv`",
        f"Quality CSV: `{output_prefix.name}_quality.csv`",
        "",
        "| portfolio | solved | avg runtime sec | avg move gap | QUBO selected |",
        "|---|---:|---:|---:|---:|",
        f"| light | {light_solved}/{len(light)} | {light_runtime:.3f} | {light_gap:.3f} | {light_qubo} |",
        f"| quality | {qual_solved}/{len(quality)} | {qual_runtime:.3f} | {qual_gap:.3f} | {qual_qubo} |",
        "",
        "Selection order: final N, moves, runtime. QUBO rows are selectable only when they do not degrade the best rule candidate on final N or moves.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_fast_candidate(
    case: str,
    instance: Instance,
    *,
    beam_width: int,
    time_limit: float,
    tail_repair_expanded: int = 10_000,
    rescue: bool = False,
) -> CandidateRecord:
    config = SolverConfig(max_depth=80, beam_width=beam_width, time_limit_sec=time_limit, entropy_alpha=1.3, random_seed=37)
    result, state = run_fast_slack_beam(
        case,
        instance,
        config,
        tail_repair_n=5,
        tail_repair_expanded=tail_repair_expanded,
        tail_repair_candidates=2,
        lookahead_pool=0,
        lookahead_keep=0,
        rescue_repair_n=8 if rescue else 0,
        rescue_repair_candidates=80 if rescue else 0,
        rescue_reservoir_size=240 if rescue else 0,
    )
    family = "rescue_grid" if rescue else "fast_grid"
    variant = f"{family}_w{beam_width}_t{time_limit:g}_tail{tail_repair_expanded}"
    if rescue:
        variant += "_rescue8c80"
    result.solver = variant
    return _candidate_from_result(
        family=family,
        variant=variant,
        result=result,
        state=state,
        qubo_used=False,
        source="grid",
    )


def _run_qubo_candidate(
    case: str,
    instance: Instance,
    *,
    supplement_width: int,
    qubo_min_depth: int,
    plateau_only: bool,
    time_limit: float,
) -> CandidateRecord:
    beam_width = 120
    config = SolverConfig(
        max_depth=80,
        beam_width=beam_width,
        time_limit_sec=time_limit,
        max_qubo_vars=60,
        qubo_penalty_a=30.0,
        diversity_weight=0.08,
        qubo_num_reads=4,
        qubo_time_limit_sec=0.01,
        qubo_sa_iterations=80,
        entropy_alpha=1.3,
        random_seed=37,
    )
    result, state = run_fast_slack_qubo_beam(
        case,
        instance,
        config,
        core_width=beam_width - supplement_width,
        supplement_width=supplement_width,
        qubo_min_depth=qubo_min_depth,
        qubo_call_interval=3,
        qubo_plateau_only=plateau_only,
        tail_repair_n=5,
        tail_repair_expanded=10_000,
    )
    plateau = "plateau" if plateau_only else "any"
    variant = f"qubo_grid_w{beam_width}_s{supplement_width}_d{qubo_min_depth}_{plateau}"
    result.solver = variant
    return _candidate_from_result(
        family="qubo_grid",
        variant=variant,
        result=result,
        state=state,
        qubo_used=True,
        source="grid",
    )


def _needs_rescue(candidates: list[CandidateRecord], exact_refs: dict[str, int]) -> bool:
    usable = [candidate for candidate in candidates if candidate.family in {"fast_grid", "qubo_grid", "seed_fast", "seed_qubo"}]
    if not usable:
        return True
    best = min(usable, key=_result_key)
    if best.result.final_n != 0:
        return True
    exact = exact_refs.get(best.case)
    if exact is None or best.result.moves is None:
        return False
    return best.result.moves - exact >= 2


def _case_filter(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return [case if case.endswith("_single") else f"{case}_single" for case in args.cases]
    if args.all_cases:
        return [f"{zone}_{bay}_single" for zone, bay in discover_single_bay_cases(ROOT / "data" / "input", {"01", "26"})]
    return list(HARD_CASES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--skip-existing-seeds", action="store_true")
    parser.add_argument("--low-beam-widths", nargs="+", type=int, default=[120, 160, 200, 240])
    parser.add_argument("--low-time-limits", nargs="+", type=float, default=[6.0, 8.0])
    parser.add_argument("--qubo-supplement-widths", nargs="+", type=int, default=[12, 24, 36])
    parser.add_argument("--qubo-min-depths", nargs="+", type=int, default=[6, 8, 10])
    parser.add_argument("--qubo-time-limit", type=float, default=8.0)
    parser.add_argument("--rescue-tail-expanded", nargs="+", type=int, default=[10_000, 30_000, 60_000])
    parser.add_argument("--rescue-time-limit", type=float, default=18.0)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_prefix = args.output_prefix or ROOT / "docs" / "experiments" / f"fast_qubo_portfolio_tuning_{stamp}"
    if not output_prefix.is_absolute():
        output_prefix = ROOT / output_prefix

    exact_refs = _exact_moves_by_case()
    candidates: list[CandidateRecord] = [] if args.skip_existing_seeds else _load_seed_candidates(exact_refs)
    _write_candidates(candidates, output_prefix.with_name(output_prefix.name + "_all_candidates.csv"), exact_refs)

    target_cases = _case_filter(args)
    if not args.skip_grid:
        for case in target_cases:
            zone, bay = _parse_case(case)
            instance = _instance_from_single_bay(zone, bay)
            print(f"CASE_START {case}", flush=True)
            for beam_width in args.low_beam_widths:
                for time_limit in args.low_time_limits:
                    print(f"RUN_FAST case={case} width={beam_width} time={time_limit}", flush=True)
                    candidate = _run_fast_candidate(case, instance, beam_width=beam_width, time_limit=time_limit)
                    candidates.append(_validate_candidate(candidate, instance))
                    _write_candidates(candidates, output_prefix.with_name(output_prefix.name + "_all_candidates.csv"), exact_refs)
            for supplement_width in args.qubo_supplement_widths:
                for qubo_min_depth in args.qubo_min_depths:
                    for plateau_only in (True, False):
                        print(
                            "RUN_QUBO "
                            f"case={case} supplement={supplement_width} min_depth={qubo_min_depth} plateau={plateau_only}",
                            flush=True,
                        )
                        candidate = _run_qubo_candidate(
                            case,
                            instance,
                            supplement_width=supplement_width,
                            qubo_min_depth=qubo_min_depth,
                            plateau_only=plateau_only,
                            time_limit=args.qubo_time_limit,
                        )
                        candidates.append(_validate_candidate(candidate, instance))
                        _write_candidates(
                            candidates,
                            output_prefix.with_name(output_prefix.name + "_all_candidates.csv"),
                            exact_refs,
                        )
            case_candidates = [candidate for candidate in candidates if candidate.case == case]
            if _needs_rescue(case_candidates, exact_refs):
                for tail_expanded in args.rescue_tail_expanded:
                    print(f"RUN_RESCUE case={case} tail={tail_expanded}", flush=True)
                    candidate = _run_fast_candidate(
                        case,
                        instance,
                        beam_width=240,
                        time_limit=args.rescue_time_limit,
                        tail_repair_expanded=tail_expanded,
                        rescue=True,
                    )
                    candidates.append(_validate_candidate(candidate, instance))
                    _write_candidates(
                        candidates,
                        output_prefix.with_name(output_prefix.name + "_all_candidates.csv"),
                        exact_refs,
                    )
            print(f"CASE_DONE {case}", flush=True)

    light = _select_portfolio(candidates, include_rescue=False)
    quality = _select_portfolio(candidates, include_rescue=True)
    _write_portfolio(
        light,
        candidates,
        output_prefix.with_name(output_prefix.name + "_light.csv"),
        output_prefix.with_name(output_prefix.name + "_light_moves"),
        exact_refs,
    )
    _write_portfolio(
        quality,
        candidates,
        output_prefix.with_name(output_prefix.name + "_quality.csv"),
        output_prefix.with_name(output_prefix.name + "_quality_moves"),
        exact_refs,
    )
    _write_report(output_prefix, light, quality)
    print(f"WROTE {output_prefix.with_name(output_prefix.name + '_all_candidates.csv')}", flush=True)
    print(f"WROTE {output_prefix.with_name(output_prefix.name + '_light.csv')}", flush=True)
    print(f"WROTE {output_prefix.with_name(output_prefix.name + '_quality.csv')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
