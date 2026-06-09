"""Experiment runner for solver/instance batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..core import Instance, ProblemVariant, Solution, validate_solution
from ..features import disorder_trace, entropy_limit, max_entropy_along_moves, total_disorder_along_moves, validate_entropy_constraint
from ..mip import solve_cpmp_3d_integer_position, solve_cpmp_4d_slot_occupancy
from ..mip.hybrid_solvers import FinalStackAssignmentMipSolver, PatternColumnGenerationSolver, RollingHorizonMipSolver
from ..qubo.batch_relocation import QuboBatchRelocationRetrievalSolver
from ..qubo.path_selection import QuboRollingHorizonSolver
from ..qubo.qubo_solvers import QuboBeamSearchSolver
from ..qubo.sampler import IsingMatrixApiQuboSampler
from ..qubo.stack_assignment import StackAssignmentGuidedPremarshallingSolver
from ..solvers import (
    BeamSearchSolver,
    DiverseBeamSearchSolver,
    ExactAStarSolver,
    ExactIDAStarSolver,
    GreedyPremarshallingSolver,
    GreedyRetrievalSolver,
    RandomSolver,
    RollingHorizonSolver,
    Solver,
    SolverConfig,
)
from .instance_generator import load_instance_json
from .reports import solution_record, solution_row, write_csv, write_json


@dataclass
class ExperimentConfig:
    """Experiment runner configuration."""

    solver_names: List[str]
    instance_paths: List[str]
    max_depth: int = 200
    beam_width: int = 100
    max_candidates_per_layer: int = 1000
    candidate_limit_per_state: Optional[int] = None
    max_qubo_vars: int = 1000
    rolling_depth: int = 4
    prefix_commit_length: int = 1
    time_limit_sec: float = 3600
    qubo_penalty_a: float = 10.0
    qubo_time_limit_sec: Optional[float] = None
    qubo_sa_iterations: int = 5000
    diversity_weight: float = 1.0
    qubo_call_interval: int = 1
    qubo_min_candidates_for_call: int = 0
    entropy_alpha: float = 1.3
    mip_warm_start: bool = True
    mip_warm_start_time_limit_sec: float = 5.0
    random_seed: int = 1
    enable_visited_pruning: bool = True
    enable_canonical_hash: bool = False
    output_csv: str = "examples/results/results.csv"
    output_json: str = "examples/results/results.json"

    def solver_config(self) -> SolverConfig:
        """Convert to solver config."""

        return SolverConfig(
            max_depth=self.max_depth,
            beam_width=self.beam_width,
            max_candidates_per_layer=self.max_candidates_per_layer,
            candidate_limit_per_state=self.candidate_limit_per_state,
            max_qubo_vars=self.max_qubo_vars,
            rolling_depth=self.rolling_depth,
            prefix_commit_length=self.prefix_commit_length,
            time_limit_sec=self.time_limit_sec,
            qubo_penalty_a=self.qubo_penalty_a,
            qubo_time_limit_sec=self.qubo_time_limit_sec,
            qubo_sa_iterations=self.qubo_sa_iterations,
            diversity_weight=self.diversity_weight,
            qubo_call_interval=self.qubo_call_interval,
            qubo_min_candidates_for_call=self.qubo_min_candidates_for_call,
            entropy_alpha=self.entropy_alpha,
            mip_warm_start=self.mip_warm_start,
            mip_warm_start_time_limit_sec=self.mip_warm_start_time_limit_sec,
            random_seed=self.random_seed,
            enable_visited_pruning=self.enable_visited_pruning,
            enable_canonical_hash=self.enable_canonical_hash,
        )


class ExperimentRunner:
    """Run selected solvers on selected instances and write reports."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> tuple[list[dict], list[dict]]:
        """Execute the configured experiment."""

        rows = []
        records = []
        solver_config = self.config.solver_config()
        for path in self.config.instance_paths:
            instance = load_instance_json(path)
            for solver_name in self.config.solver_names:
                solution = self._run_solver(solver_name, instance, solver_config)
                passed, message, final_stacks = self._validate_result(instance, solution)
                rows.append(solution_row(instance, solution, passed))
                records.append(solution_record(instance, solution, final_stacks, passed, message))
        write_csv(rows, self.config.output_csv)
        write_json(records, self.config.output_json)
        return rows, records

    def _run_solver(self, name: str, instance: Instance, config: SolverConfig) -> Solution:
        try:
            if name == "mip_3d":
                if instance.variant is not ProblemVariant.PREMARSHALLING_CPMP:
                    return Solution(name, False, False, float("inf"), 0.0, [], "NOT_APPLICABLE")
                start_moves = _mip_warm_start_moves(instance, config)
                return solve_cpmp_3d_integer_position(
                    instance,
                    config.max_depth,
                    config.time_limit_sec,
                    0.0,
                    entropy_alpha=config.entropy_alpha,
                    start_moves=start_moves,
                )
            if name == "mip_4d":
                if instance.variant is not ProblemVariant.PREMARSHALLING_CPMP:
                    return Solution(name, False, False, float("inf"), 0.0, [], "NOT_APPLICABLE")
                start_moves = _mip_warm_start_moves(instance, config)
                return solve_cpmp_4d_slot_occupancy(
                    instance,
                    config.max_depth,
                    config.time_limit_sec,
                    0.0,
                    entropy_alpha=config.entropy_alpha,
                    start_moves=start_moves,
                )
            solver = self._make_solver(name)
            return solver.solve(instance, config)
        except Exception as exc:  # noqa: BLE001
            return Solution(name, False, False, float("inf"), 0.0, [], f"ERROR: {exc}")

    def _validate_result(self, instance: Instance, solution: Solution) -> tuple[bool, str, list[list[int]]]:
        if not solution.feasible:
            return False, solution.status, instance.stacks
        passed, message, final_stacks = validate_solution(instance, solution)
        if not passed:
            return passed, message, final_stacks
        if instance.variant is ProblemVariant.PREMARSHALLING_CPMP:
            solution.entropy_alpha = self.config.entropy_alpha
            solution.entropy_limit = float(entropy_limit(instance, self.config.entropy_alpha))
            solution.entropy_max = float(max_entropy_along_moves(instance, solution.moves))
            disorder_values = disorder_trace(instance, solution.moves)
            solution.disorder_initial = float(disorder_values[0])
            solution.disorder_final = float(disorder_values[-1])
            solution.disorder_total = float(total_disorder_along_moves(instance, solution.moves))
            entropy_ok, entropy_message, _ = validate_entropy_constraint(instance, solution.moves, self.config.entropy_alpha)
            if not entropy_ok:
                return False, entropy_message, final_stacks
        return True, message, final_stacks

    @staticmethod
    def _make_solver(name: str) -> Solver:
        solvers = {
            "random": RandomSolver,
            "astar": ExactAStarSolver,
            "exact_astar": ExactAStarSolver,
            "idastar": ExactIDAStarSolver,
            "exact_idastar": ExactIDAStarSolver,
            "greedy_retrieval": GreedyRetrievalSolver,
            "greedy_cpmp": GreedyPremarshallingSolver,
            "beam": BeamSearchSolver,
            "diverse_beam": DiverseBeamSearchSolver,
            "rolling_horizon": RollingHorizonSolver,
            "qubo_beam": QuboBeamSearchSolver,
            "qubo_beam_api": _api_qubo_beam_solver,
            "qubo_rolling_horizon": QuboRollingHorizonSolver,
            "qubo_batch_relocation": QuboBatchRelocationRetrievalSolver,
            "stack_assignment_guided_cpmp": StackAssignmentGuidedPremarshallingSolver,
            "rolling_horizon_mip": RollingHorizonMipSolver,
            "final_assignment_mip": FinalStackAssignmentMipSolver,
            "pattern_column_mip": PatternColumnGenerationSolver,
        }
        if name not in solvers:
            raise ValueError(f"unknown solver: {name}")
        return solvers[name]()


def _api_qubo_beam_solver() -> QuboBeamSearchSolver:
    """Create a QUBO beam solver backed by the sibling Ising matrix API."""

    solver = QuboBeamSearchSolver(IsingMatrixApiQuboSampler(), elite_keep_ratio=0.30)
    solver.name = "qubo_beam_api"
    return solver


def _mip_warm_start_moves(instance: Instance, config: SolverConfig):
    """Find a short validated CPMP incumbent for Gurobi MIP starts."""

    if not config.mip_warm_start:
        return None
    warm_cfg = SolverConfig(
        max_depth=config.max_depth,
        beam_width=max(1, min(config.beam_width, 300)),
        max_candidates_per_layer=max(1, min(config.max_candidates_per_layer, 5000)),
        candidate_limit_per_state=config.candidate_limit_per_state,
        time_limit_sec=max(0.001, min(config.time_limit_sec, config.mip_warm_start_time_limit_sec)),
        entropy_alpha=config.entropy_alpha,
        random_seed=config.random_seed,
        enable_visited_pruning=config.enable_visited_pruning,
        enable_canonical_hash=config.enable_canonical_hash,
        score_weights=config.score_weights,
        mip_warm_start=False,
    )
    sol = BeamSearchSolver().solve(instance, warm_cfg)
    passed, _, _ = validate_solution(instance, sol) if sol.feasible else (False, "", [])
    return sol.moves if passed else None


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config JSON file."""

    import json

    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    base = Path(path).resolve().parent
    instance_paths = [
        str((base / p).resolve()) if not Path(p).is_absolute() else p
        for p in data.get("instance_paths", [])
    ]
    data["instance_paths"] = instance_paths
    return ExperimentConfig(**data)
