"""Opt-in union-selector instrumentation and static replay, never a default policy."""
from __future__ import annotations

import argparse
import copy
import itertools
import math
import random
import time
from dataclasses import asdict

import benchmark_compact_beam_qubo_n0 as beam
from container_reshuffle.qubo.model import QuboModel
from container_reshuffle.qubo.sampler import QuboSample, QuboSampleSet, QuboSamplerConfig, SimulatedAnnealingQuboSampler

PAIR_KEYS = [
    "union_diversity_qubo_same_parent_penalty",
    "union_diversity_qubo_same_source_penalty",
    "union_diversity_qubo_signature_penalty",
    "union_diversity_qubo_reduce_signature_penalty",
    "union_diversity_qubo_n_band_penalty",
    "union_diversity_qubo_origin_similarity_penalty",
    "union_diversity_qubo_same_origin_parent_n_band_penalty",
    "union_diversity_qubo_frg_witness_signature_penalty",
    "union_diversity_qubo_basin_similarity_penalty",
    "quality_reservoir_pair_receiver_debt_penalty",
    "quality_reservoir_pair_macro_signature_penalty",
    "union_diversity_qubo_unified_macro_pair_penalty",
    "union_diversity_qubo_macro_alt_same_parent_pair_penalty",
    "beam_feedback_pair_signature_penalty",
]
ARMS = ("sa", "linear", "marginal", "random")


def encode(value):
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if hasattr(value, "_asdict"):
        return encode(value._asdict())
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def decode_feature(value):
    data = dict(value)
    for key, item in data.items():
        if isinstance(item, dict) and "bytes_hex" in item:
            data[key] = bytes.fromhex(item["bytes_hex"])
        elif isinstance(item, list):
            data[key] = tuple(item)
    return beam.UnionDiversityQuboFeatures(**data)


def objective(selected, linear, features, args):
    return beam._union_diversity_objective(set(selected), linear, features, args)


def initial_set(arm, linear, features, args, target, seed):
    if arm == "linear":
        return set(sorted(range(len(linear)), key=lambda i: (linear[i], i))[:target])
    if arm == "random":
        return set(random.Random(seed).sample(range(len(linear)), target))
    if arm == "marginal":
        # Use the SAME fixed-target QUBO objective even for partial sets.
        fixed_args = copy.copy(args)
        quota = beam._union_diversity_adaptive_macro_quota_target(features, target, args)
        penalty = float(getattr(args, "union_diversity_qubo_macro_quota_penalty", 0.0))
        enabled = bool(getattr(args, "union_diversity_qubo_adaptive_macro_quota_enabled", False))
        fixed_args.union_diversity_qubo_adaptive_macro_quota_enabled = False
        def fixed_value(picked):
            count = sum(beam._union_diversity_is_macro_feature(features[i]) for i in picked)
            return objective(picked, linear, features, fixed_args) + (penalty * (count - quota) ** 2 if enabled else 0.0)
        picked = set()
        while len(picked) < target:
            picked.add(min((i for i in range(len(linear)) if i not in picked),
                           key=lambda i: (fixed_value(picked | {i}), linear[i], i)))
        return picked
    raise ValueError(arm)


class SelectorObserver:
    """Record actual production stages; substitute only the initial selection."""
    def __init__(self, arm="sa", capture_limit=0):
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.capture_limit = capture_limit
        self.records = []
        self.snapshots = []
        self.layer_context = None

    def context(self, ranked, core, best_n):
        # References survive until finish; serialization occurs AFTER solver timing.
        self.layer_context = (ranked, core, best_n)

    def sample(self, model, config, usable, target, linear, features, args, feature_sec, build_sec):
        started = time.perf_counter()
        self.current = dict(call=len(self.records), n=len(usable), k=target,
                            feature_sec=feature_sec, build_sec=build_sec, arm=self.arm)
        if self.arm == "sa":
            sample_set = SimulatedAnnealingQuboSampler().sample(model, config)
        else:
            picked = initial_set(self.arm, linear, features, args, target, config.random_seed)
            values = {i: int(i in picked) for i in range(len(usable))}
            sample_set = QuboSampleSet([QuboSample(values, model.evaluate(values))],
                                      self.arm, time.perf_counter() - started)
        self.current["initialization_sec"] = time.perf_counter() - started
        self.current["sampler_reported_sec"] = sample_set.runtime_sec
        self.current["sample_count"] = len(sample_set.samples)
        self.current["all_sample_cardinalities"] = [sum(s.binary_values.values()) for s in sample_set.samples]
        self.current["seed"] = config.random_seed
        self.pending = None
        if len(self.snapshots) < self.capture_limit:
            # Shallow-copy configuration scalars, immutable feature tuples and model.
            # No JSON or file I/O occurs inside the measured solver.
            settings = {k: v for k, v in vars(args).items()
                        if not k.startswith("_") and isinstance(v, (str, int, float, bool, type(None)))}
            settings["_quality_reservoir_objective_telemetry_enabled"] = False
            self.pending = dict(schema="union-selector-static/v1", usable=usable,
                                features=features, linear=linear, model=model.to_dict(),
                                sampler_config=asdict(config), settings=settings,
                                target=target, context=self.layer_context,
                                raw_samples=[asdict(s) for s in sample_set.samples])
        return sample_set

    def start_repair(self, picked):
        self.current["raw_selected"] = sorted(picked)
        self.current["raw_count_error"] = abs(len(picked) - self.current["k"])
        self.stage_started = time.perf_counter()

    def repair_stage(self, stage, picked):
        now = time.perf_counter()
        self.current[stage + "_sec"] = now - self.stage_started
        self.current[stage + "_selected"] = sorted(picked)
        self.stage_started = now

    def finish(self, ordered, selector_sec):
        self.current["selected"] = ordered
        self.current["selector_sec"] = selector_sec
        self.records.append(self.current)
        if self.pending is not None:
            self.pending["stages"] = self.current
            self.snapshots.append(self.pending)


def restore(snapshot):
    features = [decode_feature(f) for f in snapshot["features"]]
    args = argparse.Namespace(**snapshot["settings"])
    return list(snapshot["linear"]), features, args


def replay(snapshot, arm, seed=None, ablation="full"):
    linear, features, args = restore(snapshot)
    target = snapshot["target"]
    if ablation == "linear_only":
        for key in PAIR_KEYS:
            setattr(args, key, 0.0)
        args.union_diversity_qubo_adaptive_macro_quota_enabled = False
    elif ablation == "no_quota":
        args.union_diversity_qubo_adaptive_macro_quota_enabled = False
    elif ablation.startswith("no_"):
        key = "union_diversity_qubo_" + ablation[3:] + "_penalty"
        if key not in PAIR_KEYS:
            raise ValueError(ablation)
        setattr(args, key, 0.0)
    elif ablation != "full":
        raise ValueError(ablation)
    config = QuboSamplerConfig(**snapshot["sampler_config"])
    if seed is not None:
        config.random_seed = seed
    started = time.perf_counter()
    if arm == "sa":
        if ablation != "full":
            raise ValueError("Ablations use marginal initialization, not an unmodified full QUBO")
        samples = SimulatedAnnealingQuboSampler().sample(QuboModel.from_dict(snapshot["model"]), config)
        picked = {i for i, bit in samples.samples[0].binary_values.items() if bit}
    else:
        picked = initial_set(arm, linear, features, args, target, config.random_seed)
    initial_sec = time.perf_counter() - started
    row = dict(arm=arm, ablation=ablation, n=len(linear), k=target,
               seed=config.random_seed, raw_count_error=abs(len(picked) - target),
               initial_sec=initial_sec, raw_objective=objective(picked, linear, features, args))
    stages = {}
    stage_started = time.perf_counter()
    def stage(name, selected):
        nonlocal stage_started
        now = time.perf_counter()
        stages[name] = sorted(selected)
        row[name + "_sec"] = now - stage_started
        row[name + "_objective"] = objective(selected, linear, features, args)
        stage_started = time.perf_counter()
    selected = beam._repair_union_diversity_selection(picked, target, linear, features, args, stage_callback=stage)
    row["selected"] = sorted(selected)
    row["total_sec"] = initial_sec + row["cardinality_sec"] + row["swap_stable_sec"]
    row["swap_gain"] = row["cardinality_objective"] - row["swap_stable_objective"]
    row["one_swap_stable"] = True  # production repair exits only after a complete stable scan
    # Cross-evaluate every ablation against the original full objective.
    full_linear, full_features, full_args = restore(snapshot)
    row["full_objective"] = objective(selected, full_linear, full_features, full_args)
    row["linear_sum"] = sum(linear[i] for i in selected)
    row["pair_and_quota"] = row["swap_stable_objective"] - row["linear_sum"]
    return row


def exact_reference(snapshot, max_combinations=300000):
    """Enumerate ALL exact-k subsets if bounded; otherwise explicitly decline."""
    import numpy as np
    n, k = len(snapshot["linear"]), snapshot["target"]
    count = math.comb(n, k)
    if count > max_combinations:
        return dict(status="NOT_RUN_SIZE_LIMIT", combinations=count)
    started = time.perf_counter()
    model = QuboModel.from_dict(snapshot["model"])
    linear, features, args = restore(snapshot)
    subsets = np.array(list(itertools.combinations(range(n), k)), dtype=np.int64).reshape(count, k)
    diagonal = np.array([model.Q.get((i, i), 0.0) for i in range(n)])
    pair = np.zeros((n, n))
    for (i, j), v in model.Q.items():
        if i != j:
            pair[i, j] = pair[j, i] = v
    energies = np.full(count, model.offset) + diagonal[subsets].sum(axis=1)
    for i in range(k):
        for j in range(i + 1, k):
            energies += pair[subsets[:, i], subsets[:, j]]
    best = subsets[int(np.argmin(energies))].tolist()
    return dict(status="OPTIMAL_EXHAUSTIVE_EXACT_K", combinations=count, selected=best,
                objective=objective(best, linear, features, args), runtime_sec=time.perf_counter() - started)


def audit_snapshot(snapshot):
    linear, features, args = restore(snapshot)
    model = QuboModel.from_dict(snapshot["model"])
    n, k = len(linear), snapshot["target"]
    penalty = float(getattr(args, "qubo_penalty_a", 30.0) or 30.0)
    quota = beam._union_diversity_adaptive_macro_quota_target(features, k, args)
    qp = float(getattr(args, "union_diversity_qubo_macro_quota_penalty", 0.0))
    if not getattr(args, "union_diversity_qubo_adaptive_macro_quota_enabled", False):
        qp = 0.0
    # Matrix omits quota constant, but includes cardinality constant.
    omitted_constant = qp * quota * quota
    rng = random.Random(901)
    exact_k_errors = []
    off_k_errors = []
    for size in sorted(set([0, 1, max(0, k - 1), k, min(n, k + 1), n])):
        for _ in range(10):
            picked = set(rng.sample(range(n), size))
            bits = {i: int(i in picked) for i in range(n)}
            delta = model.evaluate(bits) + omitted_constant - penalty * (size - k) ** 2 - objective(picked, linear, features, args)
            (exact_k_errors if size == k else off_k_errors).append(abs(delta))
    rerun = replay(snapshot, "sa" if snapshot["stages"]["arm"] == "sa" else snapshot["stages"]["arm"])
    return dict(exact_k_max_error=max(exact_k_errors, default=0),
                off_k_max_error=max(off_k_errors, default=0), quota_target=quota,
                selected_reproduced=rerun["selected"] == sorted(snapshot["stages"]["selected"]))
