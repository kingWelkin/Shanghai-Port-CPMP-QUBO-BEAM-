"""CSV/JSON report helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from ..core import Instance, Solution, move_to_dict


def solution_row(instance: Instance, solution: Solution, validation_passed: bool) -> Dict[str, Any]:
    """Flatten a solution into a CSV row."""

    occupancy = instance.container_count / (instance.stack_count * instance.max_height)
    return {
        "instance_id": instance.instance_id,
        "variant": instance.variant.value,
        "solver": solution.solver_name,
        "n": instance.container_count,
        "m": instance.stack_count,
        "H": instance.max_height,
        "occupancy": f"{occupancy:.6f}",
        "feasible": solution.feasible,
        "optimal": solution.optimal,
        "objective": solution.objective,
        "runtime_sec": solution.runtime_sec,
        "expanded_states": solution.expanded_states,
        "generated_states": solution.generated_states,
        "visited_states": solution.visited_states,
        "qubo_calls": solution.qubo_calls,
        "qubo_total_vars": solution.qubo_total_vars,
        "qubo_total_time_sec": solution.qubo_total_time_sec,
        "qubo_repair_count": solution.qubo_repair_count,
        "selected_avg_similarity": solution.selected_avg_similarity,
        "mip_node_count": solution.mip_node_count,
        "mip_best_bound": solution.mip_best_bound,
        "mip_gap": solution.mip_gap,
        "mip_var_count": solution.mip_var_count,
        "mip_constr_count": solution.mip_constr_count,
        "entropy_max": solution.entropy_max,
        "entropy_limit": solution.entropy_limit,
        "entropy_alpha": solution.entropy_alpha,
        "disorder_total": solution.disorder_total,
        "disorder_initial": solution.disorder_initial,
        "disorder_final": solution.disorder_final,
        "status": solution.status,
        "validation_passed": validation_passed,
    }


def write_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    """Write experiment CSV."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(records: List[Dict[str, Any]], path: str | Path) -> None:
    """Write experiment JSON."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(records, indent=2), encoding="utf-8")


def solution_record(instance: Instance, solution: Solution, final_stacks, validation_passed: bool, validation_message: str) -> Dict[str, Any]:
    """Build a detailed JSON result record."""

    return {
        "instance": {
            "instance_id": instance.instance_id,
            "variant": instance.variant.value,
            "stack_count": instance.stack_count,
            "max_height": instance.max_height,
            "container_count": instance.container_count,
            "stacks": instance.stacks,
        },
        "solver": solution.solver_name,
        "solution": {
            "feasible": solution.feasible,
            "optimal": solution.optimal,
            "objective": solution.objective,
            "runtime_sec": solution.runtime_sec,
            "status": solution.status,
            "moves": [move_to_dict(move) for move in solution.moves],
        },
        "final_stacks": final_stacks,
        "validation_passed": validation_passed,
        "validation_message": validation_message,
        "diagnostics": {
            "expanded_states": solution.expanded_states,
            "generated_states": solution.generated_states,
            "visited_states": solution.visited_states,
            "qubo_calls": solution.qubo_calls,
            "qubo_total_vars": solution.qubo_total_vars,
            "qubo_total_time_sec": solution.qubo_total_time_sec,
            "qubo_repair_count": solution.qubo_repair_count,
            "selected_avg_similarity": solution.selected_avg_similarity,
            "mip_node_count": solution.mip_node_count,
            "mip_best_bound": solution.mip_best_bound,
            "mip_gap": solution.mip_gap,
            "mip_var_count": solution.mip_var_count,
            "mip_constr_count": solution.mip_constr_count,
            "entropy_max": solution.entropy_max,
            "entropy_limit": solution.entropy_limit,
            "entropy_alpha": solution.entropy_alpha,
            "disorder_total": solution.disorder_total,
            "disorder_initial": solution.disorder_initial,
            "disorder_final": solution.disorder_final,
        },
    }
