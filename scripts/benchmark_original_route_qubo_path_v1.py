"""Original single-bay reshuffling as a state-graph path QUBO.

This experiment builds a layered graph of legal original moves up to a fixed
horizon, then asks a QUBO to select one root-to-N=0 path.  It is deliberately
different from layout-guided or beam-supplement QUBO: the binary variables are
path edges in the original move graph.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_explainable_rule_macro_beam import _write_explain_csv  # noqa: E402
from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from benchmark_fast_qubo_integrated_v1 import _exact_moves_by_case, _move_gap, _parse_case, _rel  # noqa: E402
from benchmark_target_astar_memory_n0 import _codes_to_moves  # noqa: E402
from container_reshuffle.core import Solution, apply_move, create_initial_state, validate_solution  # noqa: E402
from container_reshuffle.features import disorder_box_count, entropy_limit, max_entropy_along_moves  # noqa: E402
from container_reshuffle.qubo.model import QuboModel  # noqa: E402
from container_reshuffle.qubo.sampler import (  # noqa: E402
    ExternalPythonDimodSampler,
    QuboSample,
    QuboSampleSet,
    QuboSamplerConfig,
    SimulatedAnnealingQuboSampler,
)
from run_business_exact_target_search_sqlite import _encode, _remaining_lower_bound, _successors_with_n  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


SOLVER_NAME = "original_route_qubo_path_v1"


@dataclass
class GraphNode:
    id: int
    depth: int
    code: bytes
    n_value: int
    incoming: list[int] = field(default_factory=list)
    outgoing: list[int] = field(default_factory=list)
    sink_edge: int | None = None


@dataclass
class GraphEdge:
    id: int
    from_node: int
    to_node: int | None
    is_sink: bool = False


@dataclass
class RouteQuboRun:
    case: str
    status: str
    moves_count: int | None
    final_n: int
    runtime_sec: float
    build_sec: float
    sample_sec: float
    decode_sec: float
    graph_nodes: int
    graph_edges: int
    sink_edges: int
    qvars: int
    qterms: int
    energy: float | None
    selected_edges: int
    path_edges: int
    violation_count: int
    goal_depths: str
    moves: list


def _add_square_penalty(model: QuboModel, terms: list[tuple[int, float]], constant: float, penalty: float) -> None:
    model.offset += penalty * constant * constant
    for idx, (var, coeff) in enumerate(terms):
        model.add_linear(var, penalty * (coeff * coeff + 2.0 * constant * coeff))
        for other, other_coeff in terms[idx + 1 :]:
            model.add_quadratic(var, other, penalty * 2.0 * coeff * other_coeff)


def _case_filter(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return [case if case.endswith("_single") else f"{case}_single" for case in args.cases]
    cases = discover_single_bay_cases()
    bays = set(args.bays)
    return [case for case in cases if _parse_case(case)[1] in bays]


def _add_edge(nodes: list[GraphNode], edges: list[GraphEdge], src: int, dst: int | None, is_sink: bool) -> int:
    edge_id = len(edges)
    edges.append(GraphEdge(edge_id, src, dst, is_sink))
    nodes[src].outgoing.append(edge_id)
    if is_sink:
        nodes[src].sink_edge = edge_id
    elif dst is not None:
        nodes[dst].incoming.append(edge_id)
    return edge_id


def _build_layered_graph(instance, args: argparse.Namespace) -> tuple[list[GraphNode], list[GraphEdge], bool]:
    initial, n0 = _encode(instance.stacks, instance.stack_count, instance.max_height)
    cap = entropy_limit(instance, args.alpha)
    depth_cap = min(cap, args.horizon)
    nodes = [GraphNode(0, 0, initial, n0)]
    layers: list[dict[bytes, int]] = [{initial: 0}]
    edges: list[GraphEdge] = []
    truncated = False

    if n0 <= args.target_final_n:
        _add_edge(nodes, edges, 0, None, True)

    for depth in range(depth_cap):
        next_layer: dict[bytes, int] = {}
        for node_id in list(layers[depth].values()):
            node = nodes[node_id]
            if node.n_value <= args.target_final_n:
                continue
            if _remaining_lower_bound(node.n_value, depth, depth_cap) > args.target_final_n:
                continue
            successors = list(_successors_with_n(
                node.code,
                node.n_value,
                instance.stack_count,
                instance.max_height,
            ))
            successors.sort(key=lambda item: (item[1], item[0]))
            if args.successor_limit_per_node > 0:
                successors = successors[: args.successor_limit_per_node]
            for nxt, nxt_n in successors:
                next_depth = depth + 1
                if next_depth + nxt_n > cap:
                    continue
                if _remaining_lower_bound(nxt_n, next_depth, depth_cap) > args.target_final_n:
                    continue
                dst = next_layer.get(nxt)
                if dst is None:
                    if args.max_nodes > 0 and len(nodes) >= args.max_nodes:
                        truncated = True
                        continue
                    dst = len(nodes)
                    next_layer[nxt] = dst
                    nodes.append(GraphNode(dst, next_depth, nxt, nxt_n))
                    if nxt_n <= args.target_final_n:
                        if args.max_edges > 0 and len(edges) >= args.max_edges:
                            truncated = True
                        else:
                            _add_edge(nodes, edges, dst, None, True)
                if args.max_edges > 0 and len(edges) >= args.max_edges:
                    truncated = True
                    continue
                _add_edge(nodes, edges, node_id, dst, False)
        layers.append(next_layer)
        if truncated and args.stop_on_truncate:
            break
        if not next_layer:
            break
    return nodes, edges, truncated


def _build_path_qubo(nodes: list[GraphNode], edges: list[GraphEdge], args: argparse.Namespace) -> tuple[QuboModel, dict[int, int]]:
    model = QuboModel()
    var_for = {edge.id: model.add_binary_var(f"e_{edge.id}") for edge in edges}

    for edge in edges:
        if not edge.is_sink:
            model.add_linear(var_for[edge.id], args.move_weight)
        else:
            model.add_linear(var_for[edge.id], args.sink_weight)

    root_out = [var_for[eid] for eid in nodes[0].outgoing]
    if root_out:
        model.add_one_hot_penalty(root_out, args.flow_penalty)

    sink_vars = [var_for[edge.id] for edge in edges if edge.is_sink]
    if sink_vars:
        model.add_one_hot_penalty(sink_vars, args.terminal_penalty)

    for node in nodes[1:]:
        incoming = [(var_for[eid], 1.0) for eid in node.incoming]
        outgoing = [(var_for[eid], -1.0) for eid in node.outgoing]
        if incoming or outgoing:
            _add_square_penalty(model, incoming + outgoing, 0.0, args.flow_penalty)
    return model, var_for


def _decode_path(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    var_for: dict[int, int],
    sample: QuboSample | None,
) -> tuple[list[bytes], int, int, int]:
    if sample is None:
        return [], 0, 0, 1
    selected = {edge_id for edge_id, var in var_for.items() if sample.binary_values.get(var, 0) == 1}
    current = 0
    codes = [nodes[current].code]
    used: set[int] = set()
    violations = 0

    while True:
        chosen = [eid for eid in nodes[current].outgoing if eid in selected]
        if len(chosen) != 1:
            violations += abs(1 - len(chosen))
            return [], len(selected), len(used), violations
        eid = chosen[0]
        used.add(eid)
        edge = edges[eid]
        if edge.is_sink:
            extra = len(selected - used)
            return codes, len(selected), len(used), violations + extra
        if edge.to_node is None:
            violations += 1
            return [], len(selected), len(used), violations
        current = edge.to_node
        codes.append(nodes[current].code)
        if len(codes) > len(nodes) + 1:
            return [], len(selected), len(used), violations + 1


def _sample_model(model: QuboModel, args: argparse.Namespace) -> QuboSampleSet:
    config = QuboSamplerConfig(
        num_reads=args.qubo_reads,
        random_seed=args.seed,
        time_limit_sec=args.qubo_time_limit,
        sa_iterations=args.qubo_iterations,
        initial_temperature=args.initial_temperature,
        final_temperature=args.final_temperature,
    )
    if args.sampler == "internal":
        return SimulatedAnnealingQuboSampler().sample(model, config)
    if args.sampler == "dimod":
        return ExternalPythonDimodSampler().sample(model, config)
    if args.sampler == "neal":
        start = time.perf_counter()
        try:
            import dimod  # type: ignore
            import neal  # type: ignore

            bqm = dimod.BinaryQuadraticModel({}, {}, model.offset, dimod.BINARY)
            for (i, j), coeff in model.Q.items():
                if i == j:
                    bqm.add_variable(i, coeff)
                else:
                    bqm.add_interaction(i, j, coeff)
            ss = neal.SimulatedAnnealingSampler().sample(
                bqm,
                num_reads=max(1, args.qubo_reads),
                num_sweeps=max(1, args.qubo_iterations),
                seed=args.seed,
                beta_range=[1.0 / max(args.initial_temperature, 1e-9), 1.0 / max(args.final_temperature, 1e-9)],
            )
            samples = [
                QuboSample({int(k): int(v) for k, v in row.sample.items()}, float(row.energy))
                for row in ss.data(["sample", "energy"])
            ]
            samples.sort(key=lambda sample: sample.energy)
            return QuboSampleSet(samples=samples, sampler_name="neal", runtime_sec=time.perf_counter() - start)
        except Exception:
            fallback = SimulatedAnnealingQuboSampler().sample(model, config)
            fallback.sampler_name = "neal_fallback_internal"
            return fallback
    raise ValueError(f"unknown sampler {args.sampler}")


def _run_case(case: str, zone: str, bay: str, args: argparse.Namespace) -> RouteQuboRun:
    start = time.perf_counter()
    instance = _instance_from_single_bay(zone, bay)
    build_start = time.perf_counter()
    nodes, edges, truncated = _build_layered_graph(instance, args)
    model, var_for = _build_path_qubo(nodes, edges, args)
    build_sec = time.perf_counter() - build_start

    sink_edges = sum(1 for edge in edges if edge.is_sink)
    goal_depths = sorted({nodes[edge.from_node].depth for edge in edges if edge.is_sink})
    if not sink_edges:
        return RouteQuboRun(
            case=case,
            status="NO_GOAL_IN_GRAPH" + ("; truncated=True" if truncated else ""),
            moves_count=None,
            final_n=disorder_box_count(instance, create_initial_state(instance)),
            runtime_sec=time.perf_counter() - start,
            build_sec=build_sec,
            sample_sec=0.0,
            decode_sec=0.0,
            graph_nodes=len(nodes),
            graph_edges=len(edges),
            sink_edges=sink_edges,
            qvars=len(model.var_index),
            qterms=len(model.Q),
            energy=None,
            selected_edges=0,
            path_edges=0,
            violation_count=1,
            goal_depths="",
            moves=[],
        )

    sample_set = _sample_model(model, args)
    sample = sample_set.samples[0] if sample_set.samples else None

    decode_start = time.perf_counter()
    codes, selected_edges, path_edges, violations = _decode_path(nodes, edges, var_for, sample)
    moves = _codes_to_moves(zone, bay, codes) if codes else []
    state = create_initial_state(instance)
    final_n = disorder_box_count(instance, state)
    status = "QUBO_PATH_INFEASIBLE"
    try:
        for move in moves:
            state = apply_move(instance, state, move)
        final_n = disorder_box_count(instance, state)
        if moves and final_n <= args.target_final_n and violations == 0:
            status = "TARGET_REACHABLE_BY_QUBO_PATH"
    except Exception as exc:  # noqa: BLE001
        status = f"DECODED_PATH_INVALID; {type(exc).__name__}: {exc}"
    decode_sec = time.perf_counter() - decode_start

    if truncated:
        status += "; graph_truncated=True"
    return RouteQuboRun(
        case=case,
        status=status,
        moves_count=len(moves) if moves else None,
        final_n=final_n,
        runtime_sec=time.perf_counter() - start,
        build_sec=build_sec,
        sample_sec=sample_set.runtime_sec,
        decode_sec=decode_sec,
        graph_nodes=len(nodes),
        graph_edges=len(edges),
        sink_edges=sink_edges,
        qvars=len(model.var_index),
        qterms=len(model.Q),
        energy=sample.energy if sample else None,
        selected_edges=selected_edges,
        path_edges=path_edges,
        violation_count=violations,
        goal_depths=";".join(str(depth) for depth in goal_depths),
        moves=moves,
    )


def _validate_run(run: RouteQuboRun, zone: str, bay: str) -> tuple[bool, bool, str]:
    if not run.moves:
        return False, False, run.status
    instance = _instance_from_single_bay(zone, bay)
    solution = Solution(
        solver_name=SOLVER_NAME,
        feasible=True,
        optimal=False,
        objective=float(len(run.moves)),
        runtime_sec=run.runtime_sec,
        moves=run.moves,
        status=run.status,
    )
    goal_ok, goal_msg, _ = validate_solution(instance, solution)
    business_ok = max_entropy_along_moves(instance, run.moves) <= entropy_limit(instance, 1.3)
    return business_ok, goal_ok, goal_msg


def _write_outputs(
    runs: list[RouteQuboRun],
    output: Path,
    acceptance: Path,
    moves_paths: dict[str, Path],
    validation: dict[str, tuple[bool, bool, str]],
    exact_refs: dict[str, int],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "case",
            "solver",
            "status",
            "final_n",
            "moves",
            "runtime_sec",
            "build_sec",
            "sample_sec",
            "decode_sec",
            "graph_nodes",
            "graph_edges",
            "sink_edges",
            "qvars",
            "qterms",
            "energy",
            "selected_edges",
            "path_edges",
            "violation_count",
            "goal_depths",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "case": run.case,
                    "solver": SOLVER_NAME,
                    "status": run.status,
                    "final_n": run.final_n,
                    "moves": run.moves_count,
                    "runtime_sec": f"{run.runtime_sec:.6f}",
                    "build_sec": f"{run.build_sec:.6f}",
                    "sample_sec": f"{run.sample_sec:.6f}",
                    "decode_sec": f"{run.decode_sec:.6f}",
                    "graph_nodes": run.graph_nodes,
                    "graph_edges": run.graph_edges,
                    "sink_edges": run.sink_edges,
                    "qvars": run.qvars,
                    "qterms": run.qterms,
                    "energy": "" if run.energy is None else f"{run.energy:.6f}",
                    "selected_edges": run.selected_edges,
                    "path_edges": run.path_edges,
                    "violation_count": run.violation_count,
                    "goal_depths": run.goal_depths,
                }
            )

    with acceptance.open("w", encoding="utf-8", newline="") as f:
        fields = [
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
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            business_ok, goal_ok, msg = validation[run.case]
            exact = exact_refs.get(run.case)
            gap = ""
            if exact is not None and run.moves_count is not None and run.final_n == 0:
                gap = run.moves_count - exact
            writer.writerow(
                {
                    "case": run.case,
                    "selected_solver": SOLVER_NAME,
                    "final_n": run.final_n,
                    "moves": run.moves_count,
                    "runtime_sec": f"{run.runtime_sec:.6f}",
                    "exact_moves": "" if exact is None else exact,
                    "move_gap": gap,
                    "qubo_used": "True",
                    "selected_reason": (
                        f"{run.status}; edge_path_qubo=True; qvars={run.qvars}; qterms={run.qterms}; "
                        f"selected_edges={run.selected_edges}; path_edges={run.path_edges}; "
                        f"violations={run.violation_count}; goal_depths={run.goal_depths}"
                    ),
                    "business_ok": str(business_ok),
                    "goal_validation_ok": str(goal_ok),
                    "validation_msg": msg,
                    "moves_path": _rel(moves_paths.get(run.case, Path(""))),
                }
            )


def _write_report(runs: list[RouteQuboRun], output: Path, acceptance: Path) -> None:
    report = output.with_suffix(".md")
    solved = sum(1 for run in runs if run.final_n == 0 and run.moves_count is not None and run.violation_count == 0)
    avg_runtime = sum(run.runtime_sec for run in runs) / len(runs) if runs else 0.0
    lines = [
        f"# {SOLVER_NAME}",
        "",
        "Original route-level QUBO over a layered legal-move state graph.",
        "",
        f"Result CSV: `{_rel(output)}`",
        f"Acceptance CSV: `{_rel(acceptance)}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| cases | {len(runs)} |",
        f"| QUBO path solved | {solved}/{len(runs)} |",
        f"| avg runtime sec | {avg_runtime:.3f} |",
        f"| max qvars | {max((run.qvars for run in runs), default=0)} |",
        f"| max qterms | {max((run.qterms for run in runs), default=0)} |",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--bays", nargs="+", default=["01"])
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "experiments" / "original_route_qubo_path_v1.csv")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--horizon-from-exact", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-final-n", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--max-nodes", type=int, default=50000)
    parser.add_argument("--max-edges", type=int, default=500000)
    parser.add_argument("--successor-limit-per-node", type=int, default=0)
    parser.add_argument("--stop-on-truncate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--move-weight", type=float, default=1.0)
    parser.add_argument("--sink-weight", type=float, default=-0.25)
    parser.add_argument("--flow-penalty", type=float, default=80.0)
    parser.add_argument("--terminal-penalty", type=float, default=80.0)
    parser.add_argument("--qubo-time-limit", type=float, default=5.0)
    parser.add_argument("--sampler", choices=["internal", "dimod", "neal"], default="neal")
    parser.add_argument("--qubo-reads", type=int, default=128)
    parser.add_argument("--qubo-iterations", type=int, default=4000)
    parser.add_argument("--initial-temperature", type=float, default=8.0)
    parser.add_argument("--final-temperature", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    acceptance = output.with_name(output.stem + "_acceptance.csv")
    moves_dir = output.with_name(output.stem + "_moves")
    moves_dir.mkdir(parents=True, exist_ok=True)
    exact_refs = _exact_moves_by_case()

    runs: list[RouteQuboRun] = []
    validation: dict[str, tuple[bool, bool, str]] = {}
    moves_paths: dict[str, Path] = {}
    cases = _case_filter(args)
    for idx, case in enumerate(cases, start=1):
        zone, bay = _parse_case(case)
        original_horizon = args.horizon
        if args.horizon_from_exact and case in exact_refs:
            args.horizon = int(exact_refs[case])
        print(f"CASE_START {idx}/{len(cases)} {case} solver={SOLVER_NAME} horizon={args.horizon}", flush=True)
        run = _run_case(case, zone, bay, args)
        args.horizon = original_horizon
        business_ok, goal_ok, msg = _validate_run(run, zone, bay)
        if run.moves:
            move_path = moves_dir / f"{case}_moves.csv"
            _write_explain_csv(_instance_from_single_bay(zone, bay), run.moves, move_path)
            moves_paths[case] = move_path
        runs.append(run)
        validation[case] = (business_ok, goal_ok, msg)
        _write_outputs(runs, output, acceptance, moves_paths, validation, exact_refs)
        _write_report(runs, output, acceptance)
        print(
            "CASE_DONE "
            f"{case} status={run.status.split(';')[0]} final_n={run.final_n} moves={run.moves_count} "
            f"runtime={run.runtime_sec:.3f}s valid={business_ok and goal_ok} "
            f"qvars={run.qvars} qterms={run.qterms} violations={run.violation_count}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
