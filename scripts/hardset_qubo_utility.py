"""Opt-in CPMP redundancy models; no solver defaults or candidate supply changes."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import itertools
import time

import benchmark_compact_beam_qubo_n0 as beam
from beam_component_ablation import ComponentObserver
from container_reshuffle.qubo.model import QuboModel
from container_reshuffle.qubo.sampler import QuboSample, QuboSampleSet, SimulatedAnnealingQuboSampler

ARMS = ('native', 'linear', 'generic_pair', 'recovery_pair', 'recovery_core', 'recovery_core_sa')
PAIR_WEIGHT = 0.35
BACKGROUND_WEIGHT = 0.25
GENERIC_MAX = 1.23
BACKGROUND_LIMIT = 32
TAIL_LIMIT = 6


@dataclass
class RouteProfile:
    structural: Counter
    tail: Counter
    cached: bool
    observed_moves: int


def stable_length(stack):
    count = 0
    for i, priority in enumerate(stack):
        if i and priority > stack[i - 1]:
            break
        count += 1
    return count


def route_profile(stacks, max_height, cached=None):
    """Preserve exact priority groups and exclude arbitrary physical stack indices."""
    structural = Counter()
    for stack in stacks:
        fixed = stable_length(stack)
        if fixed == len(stack):
            if len(stack) < max_height:
                structural[('receiver', stack[-1] if stack else 0, max_height - len(stack))] += 1
            continue
        threshold = min(stack[:fixed]) if fixed else 0
        for position in range(fixed, len(stack)):
            structural[('blocked', stack[position], threshold, min(3, len(stack) - position - 1))] += 1
        structural[('exposed_bad', stack[-1], threshold)] += 2
    tail = Counter()
    layout = [list(stack) for stack in stacks]
    observed = 0
    moves = tuple(getattr(cached, 'move_path', ())) if cached is not None else ()
    for position, move in enumerate(moves[:TAIL_LIMIT]):
        source, dest = int(move.from_stack), int(move.to_stack)
        if (source == dest or not 0 <= source < len(layout) or not 0 <= dest < len(layout)
                or not layout[source] or len(layout[dest]) >= max_height
                or layout[source][-1] != move.box):
            raise ValueError('Cached route prefix does not replay from its candidate')
        src, dst = layout[source], layout[dest]
        receiver_top = dst[-1] if dst else 0
        clean_receiver = stable_length(dst) == len(dst)
        clean_source = stable_length(src) == len(src)
        tail[('action', position, int(move.box), receiver_top, clean_source, clean_receiver)] += 1
        dst.append(src.pop())
        observed += 1
    return RouteProfile(structural, tail, bool(moves), observed)


def weighted_overlap(left, right):
    keys = left.keys() | right.keys()
    denominator = sum(max(left[k], right[k]) for k in keys)
    return sum(min(left[k], right[k]) for k in keys) / denominator if denominator else 0.0


def similarity(left, right):
    structural = weighted_overlap(left.structural, right.structural)
    # Unknown evidence does not constitute a distinct or failed continuation.
    if left.cached and right.cached:
        return 0.75 * structural + 0.25 * weighted_overlap(left.tail, right.tail)
    return structural


def profile_key(profile):
    return tuple(sorted(profile.structural.items())), tuple(sorted(profile.tail.items()))


def background_indices(size, limit=BACKGROUND_LIMIT):
    if size <= limit:
        return list(range(size))
    return [i * (size - 1) // (limit - 1) for i in range(limit)]


def background_shift(profiles, background):
    distinct = {profile_key(profile): profile for profile in background}
    result = []
    for profile in profiles:
        closest = sorted((similarity(profile, other) for other in distinct.values()), reverse=True)[:4]
        result.append(sum(closest) / len(closest) if closest else 0.0)
    return result


def generic_pair(left, right, args):
    band = max(1, int(getattr(args, 'union_diversity_qubo_n_band_size', 1)))
    predicates = {
        'same_parent': left.parent_code == right.parent_code,
        'same_source': left.source_stack >= 0 and left.source_stack == right.source_stack,
        'signature': left.signature == right.signature,
        'reduce_signature': left.reduce_signature == right.reduce_signature,
        'n_band': left.n_value // band == right.n_value // band,
    }
    return sum(float(getattr(args, 'union_diversity_qubo_' + key + '_penalty', 0.0))
               for key, enabled in predicates.items() if enabled) / GENERIC_MAX


def pair_matrices(profiles, features, args):
    size = len(profiles)
    recovery = [[0.0] * size for _ in range(size)]
    generic = [[0.0] * size for _ in range(size)]
    for i, j in itertools.combinations(range(size), 2):
        recovery[i][j] = recovery[j][i] = PAIR_WEIGHT * similarity(profiles[i], profiles[j])
        value = generic_pair(features[i], features[j], args)
        if not -1e-12 <= value <= 1 + 1e-12:
            raise ValueError('Generic control exceeds its frozen scale')
        generic[i][j] = generic[j][i] = PAIR_WEIGHT * value
    return generic, recovery


def objective(picked, linear, pair):
    return sum(linear[i] for i in picked) + sum(pair[i][j] for i, j in itertools.combinations(sorted(picked), 2))


def greedy(linear, pair, target):
    picked = set()
    while len(picked) < target:
        chosen = min((i for i in range(len(linear)) if i not in picked),
                     key=lambda i: (linear[i] + sum(pair[i][j] for j in picked), linear[i], i))
        picked.add(chosen)
    return picked


def repair(picked, linear, pair, target, callback=None):
    """Fixed objective for all partial/exact sets; exact-k and strict 1-swap descent."""
    picked = set(picked)
    while len(picked) > target:
        chosen = min(picked, key=lambda i: (-linear[i] - sum(pair[i][j] for j in picked if j != i), i))
        picked.remove(chosen)
    while len(picked) < target:
        chosen = min((i for i in range(len(linear)) if i not in picked),
                     key=lambda i: (linear[i] + sum(pair[i][j] for j in picked), linear[i], i))
        picked.add(chosen)
    if callback:
        callback('cardinality', picked)
    swaps = 0
    while True:
        best = None
        best_delta = -1e-12
        for out in sorted(picked):
            retained = picked - {out}
            removed = linear[out] + sum(pair[out][j] for j in retained)
            for incoming in range(len(linear)):
                if incoming in picked:
                    continue
                delta = linear[incoming] + sum(pair[incoming][j] for j in retained) - removed
                if delta < best_delta:
                    best = out, incoming
                    best_delta = delta
        if best is None:
            break
        picked.remove(best[0])
        picked.add(best[1])
        swaps += 1
    if callback:
        callback('swap_stable', picked)
    return picked, swaps


def qubo_model(linear, pair, target, penalty=30.0):
    model = QuboModel()
    variables = [model.add_binary_var('route_' + str(i)) for i in range(len(linear))]
    for i, value in enumerate(linear):
        model.add_linear(variables[i], value)
        for j in range(i + 1, len(linear)):
            if pair[i][j]:
                model.add_quadratic(variables[i], variables[j], pair[i][j])
    model.add_exactly_k_penalty(variables, target, penalty)
    return model


class RouteObserver(ComponentObserver):
    """Retain production supply/commit validation; replace only scoped Union choice."""
    def __init__(self, branch):
        if branch not in ARMS:
            raise ValueError(branch)
        super().__init__()
        self.branch = branch
        self.custom_linear = None
        self.custom_pair = None
        self.custom_model = None
        self.inside = False
        self.route_snapshots = []

    def install(self):
        super().install()
        for name, callback in {
            '_select_union_diversity_qubo_supplement': self.select,
            '_union_diversity_objective': self.scoped_objective,
            '_repair_union_diversity_selection': self.scoped_repair,
        }.items():
            self.originals[name] = getattr(beam, name)
            setattr(beam, name, callback)

    def select(self, *args, **kwargs):
        if self.inside:
            raise RuntimeError('Nested Union call')
        self.inside = True
        try:
            return self.originals['_select_union_diversity_qubo_supplement'](*args, **kwargs)
        finally:
            self.inside = False
            self.custom_linear = self.custom_pair = self.custom_model = None

    def scoped_objective(self, picked, linear, features, args):
        if self.inside and self.custom_linear is not None:
            if list(features) != list(self.features):
                raise ValueError('Objective called on a different candidate pool')
            return objective(picked, self.custom_linear, self.custom_pair)
        return self.originals['_union_diversity_objective'](picked, linear, features, args)

    def scoped_repair(self, picked, target, linear, features, args, stage_callback=None):
        if self.custom_linear is None:
            return self.originals['_repair_union_diversity_selection'](
                picked, target, linear, features, args, stage_callback=stage_callback)
        result, swaps = repair(picked, self.custom_linear, self.custom_pair, target, stage_callback)
        self.current['exchange_swaps'] = swaps
        return result

    def sample(self, model, config, usable, target, linear, features, args, feature_sec, build_sec):
        began = time.perf_counter()
        ranked, retained, _best_n = self.layer_context
        retained = list(retained)
        cache = getattr(args, '_union_rollout_probe_cache', {}) or {}
        def profile(code):
            return route_profile(beam._code_to_stacks(code, args._stack_count, args._max_height),
                                 args._max_height, cache.get(code))
        profiles = [profile(code) for code, _data in usable]
        background_codes = [retained[i] for i in background_indices(len(retained))]
        background = [profile(code) for code in background_codes]
        shifts = background_shift(profiles, background)
        generic, recovery = pair_matrices(profiles, features, args)
        semantic_sec = time.perf_counter() - began
        self.active_args = args
        self.usable, self.features, self.linear = list(usable), features, linear
        if self.branch == 'native':
            result = super().sample(model, config, usable, target, linear, features, args, feature_sec, build_sec)
        else:
            size = len(usable)
            self.custom_linear = [float(value) for value in linear]
            self.custom_pair = ([[0.0] * size for _ in range(size)] if self.branch == 'linear'
                                else generic if self.branch == 'generic_pair' else recovery)
            if self.branch in ('recovery_core', 'recovery_core_sa'):
                self.custom_linear = [value + BACKGROUND_WEIGHT * shift
                                      for value, shift in zip(self.custom_linear, shifts)]
            build_started = time.perf_counter()
            self.custom_model = qubo_model(self.custom_linear, self.custom_pair, target)
            custom_build_sec = time.perf_counter() - build_started
            initialized = time.perf_counter()
            if self.branch == 'recovery_core_sa':
                result = SimulatedAnnealingQuboSampler().sample(self.custom_model, config)
            else:
                selected = greedy(self.custom_linear, self.custom_pair, target)
                bits = {i: int(i in selected) for i in range(size)}
                result = QuboSampleSet([QuboSample(bits, self.custom_model.evaluate(bits))],
                                      'marginal', time.perf_counter() - initialized)
            self.current = dict(call=len(self.records), layer=self.current_layer, n=size, k=target,
                                feature_sec=feature_sec, build_sec=build_sec,
                                custom_build_sec=custom_build_sec, arm=self.branch,
                                initialization_sec=time.perf_counter() - initialized,
                                sampler_reported_sec=result.runtime_sec,
                                sample_count=len(result.samples), seed=config.random_seed,
                                all_sample_cardinalities=[sum(s.binary_values.values()) for s in result.samples],
                                macro_pool=sum(beam._union_diversity_is_macro_feature(f) for f in features))
            self.pending = self.pending_snapshot = None
        self.current.update(arm=self.branch, semantic_sec=semantic_sec, background_size=len(retained),
                            representative_count=len(background_codes),
                            distinct_background_profiles=len({profile_key(p) for p in background}),
                            background_shift_min=min(shifts, default=0.0),
                            background_shift_max=max(shifts, default=0.0),
                            recovery_pair_min=min((recovery[i][j] for i, j in itertools.combinations(range(len(profiles)), 2)), default=0.0),
                            recovery_pair_max=max((recovery[i][j] for i, j in itertools.combinations(range(len(profiles)), 2)), default=0.0),
                            cached_pool_profiles=sum(p.cached for p in profiles),
                            cached_background_profiles=sum(p.cached for p in background),
                            replayed_cached_moves=sum(p.observed_moves for p in profiles + background))
        if len(usable) > target and len(self.route_snapshots) < 2:
            self.route_snapshots.append(dict(layer=self.current_layer, target=target,
                usable=[dict(code=code.hex(), data=[data[0], data[1], data[2].hex()]) for code, data in usable],
                ordered_background=[code.hex() for code in retained],
                background_representatives=[code.hex() for code in background_codes],
                base_linear=list(linear), generic=generic, recovery=recovery, background_shift=shifts,
                profiles=[dict(structural=list(p.structural.items()), tail=list(p.tail.items()),
                               cached=p.cached, observed_moves=p.observed_moves) for p in profiles],
                stages=self.current))
        return result

    def finish(self, ordered, selector_sec):
        if self.custom_linear is not None:
            selected = set(ordered)
            if len(selected) != self.current['k']:
                raise RuntimeError('Research selection violates target count')
            bits = {i: int(i in selected) for i in range(len(self.custom_linear))}
            discrepancy = self.custom_model.evaluate(bits) - objective(selected, self.custom_linear, self.custom_pair)
            if abs(discrepancy) > 1e-8:
                raise RuntimeError('Repaired set does not match its QUBO objective')
            self.current['exact_k_objective_discrepancy'] = discrepancy
        super().finish(ordered, selector_sec)
