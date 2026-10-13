"""Freeze/run/replay a development-only Beam selector attribution protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src"):
    sys.path.insert(0, str(path))

import benchmark_compact_beam_qubo_n0 as beam
import open_cpmp_benchmarks as opened
import benchmark_compact_paper_set as paper
from beam_selector_research import ARMS, SelectorObserver, audit_snapshot, encode, exact_reference, replay

PROFILE = "scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_short_incumbent_v42_polish"
DEFAULT_OUTPUT = ROOT / "docs/experiments/beam_selector_20260922"
SEEDS = [37, 41, 43, 47, 53]
ENVIRONMENT = {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def make_args(cap, seed):
    args = beam.make_compact_args(solver_profile=PROFILE, time_limit=cap, seed=seed,
                                  alpha=10.0, business_cap_enabled=False)
    beam._apply_solver_profile(args)
    beam._apply_qubo_influence_profile(args)
    beam._apply_dynamic_profile(args)
    return args


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(output, cap):
    if (output / "baseline_manifest.json").exists():
        raise FileExistsError("Protocol is already frozen; choose another output directory to change it")
    cases = opened.discover_dlts_cases(ROOT / "benchmarks/open120", limit_per_family=1)
    records = [dict(case=c.case_id, dataset="open120", family=c.family, path=c.instance_path.relative_to(ROOT).as_posix(),
                    max_height=c.reference.max_height, sha256=digest(c.instance_path)) for c in cases]
    paper_cases = paper.discover_sampled_paper_cases(ROOT / "benchmarks/paper", datasets=["BF"],
                                                    families=["BF31", "BF32", "BF33"], sample_per_family=2)
    records.extend(dict(case=c.case_id, dataset=c.dataset, family=c.family,
                        path=c.instance_path.relative_to(ROOT).as_posix(), max_height=None,
                        sha256=digest(c.instance_path)) for c in paper_cases)
    args = make_args(cap, SEEDS[0])
    config = {k: v for k, v in vars(args).items() if not k.startswith("_") and isinstance(v, (str, int, float, bool, type(None)))}
    sources = ["scripts/benchmark_compact_beam_qubo_n0.py", "scripts/beam_selector_research.py",
               "scripts/run_beam_selector_research.py", "container_reshuffle/qubo/model.py", "container_reshuffle/qubo/sampler.py"]
    manifest = dict(schema="beam-selector-diagnostic/v1", frozen_date="2026-09-22", cases=records,
                    seeds=SEEDS, arms=list(ARMS), per_solver_cap_sec=cap, profile=PROFILE,
                    config=config, parent_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    source_sha256={p: digest(ROOT / p) for p in sources}, python=sys.version,
                    platform=platform.platform(), processor=platform.processor(), logical_cpus=os.cpu_count(),
                    dependencies={p: importlib.metadata.version(p) for p in ["numpy", "dimod", "dwave-neal", "pytest"]},
                    environment=ENVIRONMENT, jobs=1, snapshots="First two selector calls for seed 37, sa and linear trajectories",
                    scope="Development diagnostic of general v4.3 route; not routed v4.3 or Fresh30 reproduction; no promotion",
                    timing="solver call and validator measured separately; process wall also saved; no incumbent discovery time inferred",
                    success="validator-clean terminal; budget success additionally requires solver wall <= cap, no tolerance")
    write_json(output / "baseline_manifest.json", manifest)
    write_json(output / "instance_split.json", dict(development=[c["case"] for c in records],
                validation=[], final_test=[], notes="Open120 and published corpora have been exposed previously. Unused cases are not claimed unseen. Split all derivatives by original instance."))
    print(f"Frozen {len(records)} cases x {len(SEEDS)} seeds x {len(ARMS)} arms", flush=True)


def worker(output, case_id, arm, seed):
    manifest = json.loads((output / "baseline_manifest.json").read_text(encoding="utf-8"))
    case = next(c for c in manifest["cases"] if c["case"] == case_id)
    started = time.perf_counter()
    instance = (paper.load_bf_bay(ROOT / case["path"]) if case.get("dataset") == "BF"
                else opened.load_dlts_dat(ROOT / case["path"], max_height=case["max_height"]))
    load_sec = time.perf_counter() - started
    args = make_args(manifest["per_solver_cap_sec"], seed)
    observer = SelectorObserver(arm, 2 if seed == 37 and arm in ("sa", "linear") else 0)
    args._union_selector_observer = observer
    started = time.perf_counter()
    result = beam.run_compact_beam_instance(case_id, instance, args)
    solver_wall = time.perf_counter() - started
    started = time.perf_counter()
    business_ok, goal_ok, message = opened._open_validate_instance_result(result, instance, 10.0, business_cap_enabled=False)
    validation_sec = time.perf_counter() - started
    row = dict(case=case_id, family=case["family"], arm=arm, seed=seed,
               valid_terminal=bool(business_ok and goal_ok),
               within_budget=solver_wall <= manifest["per_solver_cap_sec"],
               budget_success=bool(business_ok and goal_ok and solver_wall <= manifest["per_solver_cap_sec"]),
               moves=len(result.moves) if goal_ok else None, returned_path_length=len(result.moves),
               solver_wall_sec=solver_wall, solver_reported_sec=result.runtime_sec,
               load_sec=load_sec, validation_sec=validation_sec, end_to_end_sec=load_sec+solver_wall+validation_sec,
               expanded=result.expanded, generated=result.generated, status=result.status, validation_message=message,
               union_calls=len(observer.records), all_qubo_calls=result.qubo_calls,
               union_actual_sa_calls=len(observer.records) if arm == "sa" else 0,
               returned_time_is_not_incumbent_discovery_time=True)
    for key in ("feature_sec", "build_sec", "initialization_sec", "cardinality_sec", "swap_stable_sec", "selector_sec"):
        row[key] = sum(record[key] for record in observer.records)
    row["raw_count_violations"] = sum(record["raw_count_error"] > 0 for record in observer.records)
    stem = f"{case_id}__{seed}__{arm}"
    write_json(output / "runs" / (stem + ".json"), dict(row=row, moves=[asdict(m) for m in result.moves], selector_records=observer.records))
    for index, snapshot in enumerate(observer.snapshots):
        snapshot.update(case=case_id, family=case["family"], trajectory_arm=arm, trajectory_seed=seed, snapshot_id=f"{stem}__{index}")
        write_json(output / "snapshots" / f"{stem}__{index}.json", encode(snapshot))
    print(json.dumps(row, ensure_ascii=False), flush=True)


def run(output, limit=None):
    manifest = json.loads((output / "baseline_manifest.json").read_text(encoding="utf-8"))
    for path, expected in manifest["source_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Frozen source changed: {path}; freeze into a new output before running")
    for case in manifest["cases"]:
        if digest(ROOT / case["path"]) != case["sha256"]:
            raise ValueError(f"Input changed: {case['case']}")
    schedule = []
    for ci, case in enumerate(manifest["cases"]):
        for si, seed in enumerate(manifest["seeds"]):
            offset = (ci + si) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                schedule.append((case, arm, seed))
    if limit is not None:
        schedule = schedule[:limit]
    environment = {**os.environ, **ENVIRONMENT}
    for index, (case, arm, seed) in enumerate(schedule):
        run_path = output / "runs" / f"{case['case']}__{seed}__{arm}.json"
        if run_path.exists():
            continue
        started = time.perf_counter()
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(output),
                               "--case", case["case"], "--arm", arm, "--seed", str(seed)],
                              cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace")
        process_sec = time.perf_counter() - started
        if proc.returncode:
            write_json(output / "errors" / f"{case['case']}__{seed}__{arm}.json",
                       dict(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr))
            raise RuntimeError(proc.stderr)
        data = json.loads(run_path.read_text(encoding="utf-8"))
        data["row"]["process_wall_sec"] = process_sec
        write_json(run_path, data)
        row = data["row"]
        print(f"{index+1}/{len(schedule)} {case['case']} {seed} {arm}: valid={row['valid_terminal']} moves={row['moves']} wall={row['solver_wall_sec']:.3f}s calls={row['union_calls']}", flush=True)
    rows = [json.loads(p.read_text(encoding="utf-8"))["row"] for p in sorted((output / "runs").glob("*.json"))]
    write_csv(output / "baseline_results.csv", rows)


def run_replay(output):
    attribution, ablations, audits, references = [], [], [], []
    for path in sorted((output / "snapshots").glob("*.json")):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        identity = {key: snapshot[key] for key in ("snapshot_id", "case", "family", "trajectory_arm")}
        audit = audit_snapshot(snapshot)
        audits.append({**identity, **audit})
        if audit["exact_k_max_error"] > 1e-7 or not audit["selected_reproduced"]:
            raise AssertionError({**identity, **audit})
        reference = exact_reference(snapshot)
        references.append({**identity, **reference})
        for arm in ARMS:
            for trial in range(5 if arm in ("sa", "random") else 1):
                row = replay(snapshot, arm, seed=None if trial == 0 else 800 + trial)
                if reference["status"] == "OPTIMAL_EXHAUSTIVE_EXACT_K":
                    row["proxy_optimality_gap"] = row["full_objective"] - reference["objective"]
                attribution.append({**identity, **row})
        # Four starts are reported with all work included, never as a one-start comparison.
        starts = [replay(snapshot, "random", seed=701+j) for j in range(4)]
        best = dict(min(starts, key=lambda row: row["full_objective"]))
        best.update(arm="random4", total_sec=sum(r["total_sec"] for r in starts), starts=4,
                    proxy_optimality_gap=best["full_objective"]-reference["objective"] if "objective" in reference else None)
        attribution.append({**identity, **best})
        for ablation in ("full", "linear_only", "no_same_parent", "no_same_source", "no_signature", "no_reduce_signature", "no_n_band", "no_quota"):
            ablations.append({**identity, **replay(snapshot, "marginal", ablation=ablation)})
        print(f"Replayed {path.stem}: exact-k error={audit['exact_k_max_error']:.3g} off-k error={audit['off_k_max_error']:.3g}", flush=True)
    write_csv(output / "selector_attribution.csv", attribution)
    write_csv(output / "term_ablation.csv", ablations)
    write_csv(output / "objective_audit.csv", audits)
    write_csv(output / "exact_references.csv", references)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run", "worker", "replay"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cap", type=float, default=2.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case")
    parser.add_argument("--arm", choices=ARMS, default="sa")
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.output, args.cap)
    elif args.command == "run":
        run(args.output, args.limit)
    elif args.command == "worker":
        worker(args.output, args.case, args.arm, args.seed)
    else:
        run_replay(args.output)


if __name__ == "__main__":
    main()
