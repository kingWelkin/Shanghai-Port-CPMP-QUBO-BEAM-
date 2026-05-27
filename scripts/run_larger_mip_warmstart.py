"""Run larger CPMP MIP benchmark with a validated beam MIP start."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from container_reshuffle.core import move_from_dict, validate_solution
from container_reshuffle.experiments.instance_generator import load_instance_json
from container_reshuffle.experiments.reports import solution_record, solution_row, write_csv, write_json
from container_reshuffle.features import validate_entropy_constraint
from container_reshuffle.mip.mip_3d_integer_position import solve_cpmp_3d_integer_position
from container_reshuffle.mip.mip_4d_slot_occupancy import solve_cpmp_4d_slot_occupancy


def _load_beam_moves(path: Path):
    records = json.loads(path.read_text(encoding="utf-8"))
    for record in records:
        if record.get("solver") == "beam" and record.get("solution", {}).get("feasible"):
            return [move_from_dict(row) for row in record["solution"]["moves"]]
    raise RuntimeError(f"no feasible beam solution found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["3d", "4d"], required=True)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--max-steps", type=int, default=33)
    args = parser.parse_args()

    instance = load_instance_json("examples/instances/larger_cpmp_6x5_n25_seed5.json")
    start_moves = _load_beam_moves(Path("docs/experiments/larger_cpmp_baselines.json"))
    entropy_ok, entropy_message, _ = validate_entropy_constraint(instance, start_moves, 1.3)
    if not entropy_ok:
        print(f"Skipping MIP start: {entropy_message}")
        start_moves = None

    if args.model == "3d":
        solution = solve_cpmp_3d_integer_position(
            instance,
            max_steps=args.max_steps,
            time_limit_sec=args.time_limit,
            mip_gap=0.0,
            output_flag=False,
            diagnostics_dir="docs/experiments/mip_diagnostics/larger_cpmp_3d_warm",
            start_moves=start_moves,
        )
        stem = "larger_cpmp_mip_3d_warm_10min"
    else:
        solution = solve_cpmp_4d_slot_occupancy(
            instance,
            max_steps=args.max_steps,
            time_limit_sec=args.time_limit,
            mip_gap=0.0,
            output_flag=False,
            diagnostics_dir="docs/experiments/mip_diagnostics/larger_cpmp_4d_warm",
            start_moves=start_moves,
        )
        stem = "larger_cpmp_mip_4d_warm_10min"

    passed, message, final_stacks = validate_solution(instance, solution) if solution.feasible else (False, solution.status, instance.stacks)
    write_csv([solution_row(instance, solution, passed)], f"docs/experiments/{stem}.csv")
    write_json([solution_record(instance, solution, final_stacks, passed, message)], f"docs/experiments/{stem}.json")
    print(
        json.dumps(
            {
                "solver": solution.solver_name,
                "feasible": solution.feasible,
                "optimal": solution.optimal,
                "objective": solution.objective,
                "runtime_sec": solution.runtime_sec,
                "status": solution.status,
                "best_bound": solution.mip_best_bound,
                "mip_gap": solution.mip_gap,
                "node_count": solution.mip_node_count,
                "var_count": solution.mip_var_count,
                "constr_count": solution.mip_constr_count,
                "validation_passed": passed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
