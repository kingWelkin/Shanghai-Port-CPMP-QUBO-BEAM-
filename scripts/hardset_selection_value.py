"""Real post-fill Beam exchanges, matched prefixes and primitive-work suffixes."""
from __future__ import annotations

import copy
from dataclasses import asdict
import inspect
import itertools
import json

import benchmark_compact_beam_qubo_n0 as beam
import container_reshuffle.solvers.bs_frg as frg
from beam_selector_research import encode
from capture_independent_macro_windows import IndependentObserver, root_boundary_state
from cpmp_budget_witness import audit_cached_route, adopt_common_witness
from hardset_qubo_utility import (route_profile, background_indices, background_shift,
    similarity, PAIR_WEIGHT, BACKGROUND_WEIGHT)
from production_route_provenance import solver_frame
from production_selector_replay import PrefixClock, REAL_TIME, fingerprint

PAIRS = {f'p{i}{j}': (i, j) for i, j in itertools.combinations(range(4), 2)}
HORIZONS = (30000, 120000)
VIRTUAL_SECONDS_PER_EXPANSION = 0.0001


class CaptureReady(Exception):
    pass


class MatchedClock(PrefixClock):
    def __init__(self, values=None, cutoff=None):
        super().__init__(values, cutoff)
        self.work_now = cutoff

    def __call__(self):
        return self.work_now if self.live else super().__call__()

    def continue_live(self):
        self.work_now = self.cutoff
        self.live = True


class ExpansionBudget:
    """Count primitive expansion requests, including compact cache hits, once."""
    def __init__(self, clock, args, limit, watched, verifier=None):
        self.clock, self.args, self.limit = clock, args, limit
        self.verifier = verifier
        self.originals = []
        self.cache_building = 0
        self.stats = dict(limit=limit, expansions=0, compact_uncached=0, compact_cached=0,
            frg_fill=0, frg_reduce=0, frg_single=0, served_successors=0, frg_choices=0,
            exhausted=False, watched_main_expansions={code.hex(): 0 for code in watched})

    def grant(self, kind, code=None):
        if self.verifier:
            self.verifier()
        if self.stats['expansions'] >= self.limit:
            self.stats['exhausted'] = True
            self.args.strict_search_deadline = True
            self.clock.work_now = max(self.clock.work_now, self.args._overall_search_deadline + 1.)
            return False
        self.stats['expansions'] += 1
        self.stats[kind] += 1
        self.clock.work_now = self.clock.cutoff + self.stats['expansions'] * VIRTUAL_SECONDS_PER_EXPANSION
        if code is not None and code.hex() in self.stats['watched_main_expansions']:
            frame = inspect.currentframe().f_back.f_back
            if frame and frame.f_code.co_name == 'run_compact_beam_instance':
                self.stats['watched_main_expansions'][code.hex()] += 1
        return True

    def install(self):
        uncached = beam._successors_with_n_uncached
        cached = beam._successors_with_n_cached
        def count_uncached(*args, **kwargs):
            if self.cache_building:
                yield from uncached(*args, **kwargs)
                return
            if not self.grant('compact_uncached', args[0]):
                return
            for row in uncached(*args, **kwargs):
                self.stats['served_successors'] += 1
                yield row
        def count_cached(*args, **kwargs):
            if not self.grant('compact_cached', args[0]):
                return ()
            self.cache_building += 1
            try:
                result = cached(*args, **kwargs)
            finally:
                self.cache_building -= 1
            self.stats['served_successors'] += len(result)
            return result
        count_cached.cache_info = cached.cache_info
        count_cached.cache_clear = cached.cache_clear
        self.replace(beam, '_successors_with_n_uncached', count_uncached)
        self.replace(beam, '_successors_with_n_cached', count_cached)
        for name, kind in (('_bg_choices', 'frg_fill'), ('_reduction_choices', 'frg_reduce'),
                           ('_single_move_children', 'frg_single')):
            original = getattr(frg, name)
            def wrapped(*args, _original=original, _kind=kind, **kwargs):
                if not self.grant(_kind):
                    return []
                result = _original(*args, **kwargs)
                self.stats['frg_choices'] += len(result)
                return result
            aliases = [key for key, value in vars(beam).items() if value is original]
            self.replace(frg, name, wrapped)
            for alias in aliases:
                self.replace(beam, alias, wrapped)

    def replace(self, module, name, value):
        self.originals.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    def restore(self):
        for module, name, original in reversed(self.originals):
            setattr(module, name, original)
        self.originals.clear()


def candidate_kind(code, args, loc):
    macro = beam._union_diversity_macro_origin(beam._union_diversity_candidate_origin(code, args))
    paths = loc.get('frg_macro_edge_paths', {})
    parents = getattr(args, '_frg_macro_edge_parent_by_code', {})
    actual_edge = code in paths and parents.get(code) == loc['candidates'][code][2]
    return bool(macro), bool(actual_edge)


def render_window(reference, positions, pool, indices, protected):
    if len(indices) != 2 or len(set(indices)) != 2 or any(i not in range(4) for i in indices):
        raise ValueError('Window requires two distinct identities from four')
    replacement = list(reference)
    for position, index in zip(positions, indices):
        replacement[position] = pool[index]
    if len(set(replacement)) != len(reference):
        raise ValueError('Window duplicates a retained state')
    if not set(protected) <= set(replacement):
        raise ValueError('Window removed a protected state')
    if any(a != b for i, (a, b) in enumerate(zip(reference, replacement)) if i not in positions):
        raise ValueError('Fixed backbone or order changed')
    return replacement


class ValueObserver(IndependentObserver):
    def __init__(self, clock, contract, expected=None, pair='p01', horizon=3000):
        super().__init__(clock, contract, expected)
        self.pair = pair
        self.horizon = horizon
        self.budget = None
        self.decision = None
        self.shared = None

    def eligible(self, selected_codes, args, loc):
        if self.context_layer != loc['layer_index'] or self.sample_layer != loc['layer_index']:
            return None
        protected = set(self.layer_context[1])
        selected = set(selected_codes)
        sources = getattr(args, '_frontier_trace_last_selected_sources', {})
        candidates = loc['candidates']
        bound = min(loc['cap'], loc['depth_cap'])
        for kind in ((True, True), (False, False)):
            positions = [i for i, code in enumerate(selected_codes) if code not in protected
                and candidate_kind(code, args, loc) == kind
                and (kind[0] or sources.get(code) == 'tail_fill')][-2:]
            if len(positions) != 2:
                continue
            outsiders = [code for code, data in self.layer_context[0]
                if code not in selected and candidate_kind(code, args, loc) == kind
                and data[0] > 0 and data[0] + data[1] <= bound
                and not beam._parent_chain_would_cycle(loc['parent'], code, data[2])][:2]
            if len(outsiders) == 2:
                return positions, [selected_codes[i] for i in positions] + outsiders, protected, kind
        return None

    def build_packet(self, selected_codes, args, loc, choice):
        positions, pool, protected, kind = choice
        cache = getattr(args, '_union_rollout_probe_cache', {})
        audits, certified = {}, []
        audit_codes = list(dict.fromkeys(pool + [code for code in selected_codes if code in cache]))
        for code in audit_codes:
            record, moves = audit_cached_route(loc['instance'], code, loc['candidates'][code], loc['parent'],
                loc.get('frg_macro_edge_paths', {}), getattr(args, '_frg_macro_edge_parent_by_code', {}),
                cache.get(code), loc['cap'])
            if code in pool and (not record['prefix_verified'] or record['actual_prefix_cost'] != loc['candidates'][code][1]):
                raise ValueError('Test variable does not have its recorded physical prefix')
            audits[code.hex()] = record
            if moves is not None:
                certified.append((code, moves))
        shadow = copy.copy(args)
        shadow._quality_reservoir_objective_telemetry_enabled = False
        shadow.micro_rollout_enabled = shadow.micro_rollout_trace_only = shadow.receiver_safety_enabled = False
        shadow._micro_rollout_feature_trace_by_code = {}
        shadow.union_diversity_qubo_adaptive_macro_quota_enabled = False
        indexed = {code: i for i, (code, _data) in enumerate(self.last_usable)}
        residual_ranks = {code: i for i, (code, data) in enumerate(
            (item for item in self.layer_context[0] if item[0] not in protected))}
        raw_pool = [beam._quality_reservoir_qubo_linear_energy(f, shadow) for f in self.last_features]
        minimum, maximum = min(raw_pool), max(raw_pool)
        features, linear = [], []
        for code in pool:
            if code in indexed:
                index = indexed[code]
                feature, value = self.last_features[index], self.last_linear[index]
            else:
                feature = beam._union_diversity_qubo_features(code, loc['candidates'][code],
                    residual_ranks[code], shadow, cache.get(code),
                    origin=beam._union_diversity_candidate_origin(code, shadow))
                raw = beam._quality_reservoir_qubo_linear_energy(feature, shadow)
                value = (raw - minimum) / (maximum - minimum) if maximum > minimum else 0.
            features.append(feature)
            linear.append(float(value))
        fixed_union = [self.last_features[i] for i in self.current.get('selected', [])
                       if self.last_usable[i][0] not in pool[:2]]
        def pair_value(left, right):
            return beam._union_diversity_objective({0, 1}, [0., 0.], [left, right], shadow)
        native_shift = [sum(pair_value(feature, fixed) for fixed in fixed_union) for feature in features]
        native_pair = [[0.] * 4 for _ in range(4)]
        profiles = [route_profile(beam._code_to_stacks(code, args._stack_count, args._max_height),
                                  args._max_height, cache.get(code)) for code in pool]
        backbone = [code for i, code in enumerate(selected_codes) if i not in positions]
        representatives = [backbone[i] for i in background_indices(len(backbone))]
        background = [route_profile(beam._code_to_stacks(code, args._stack_count, args._max_height),
                                    args._max_height, cache.get(code)) for code in representatives]
        shifts = background_shift(profiles, background)
        recovery_pair = [[0.] * 4 for _ in range(4)]
        for i, j in itertools.combinations(range(4), 2):
            native_pair[i][j] = native_pair[j][i] = pair_value(features[i], features[j])
            recovery_pair[i][j] = recovery_pair[j][i] = PAIR_WEIGHT * similarity(profiles[i], profiles[j])
        scores = {}
        for name, (i, j) in PAIRS.items():
            scores[name] = dict(linear=linear[i] + linear[j],
                native_q=linear[i] + linear[j] + native_shift[i] + native_shift[j] + native_pair[i][j],
                recovery_q=linear[i] + linear[j] + BACKGROUND_WEIGHT * (shifts[i] + shifts[j]) + recovery_pair[i][j])
        adoption = adopt_common_witness(solver_frame(), certified)
        available = [value for value in (adoption['old_best_cost'], adoption['new_cost']) if value is not None]
        adoption['available_best_cost'] = min(available) if available else None
        packet = dict(reference=[code.hex() for code in selected_codes], positions=positions,
            pool=[dict(code=code.hex(), data=[loc['candidates'][code][0], loc['candidates'][code][1], loc['candidates'][code][2].hex()],
                       origin=beam._union_diversity_candidate_origin(code, args), in_native_pool=code in indexed) for code in pool],
            protected=[code.hex() for code in sorted(protected)], kind='macro' if kind[0] else 'ordinary',
            macro_count=sum(candidate_kind(code, args, loc)[0] for code in selected_codes),
            edge_count=sum(candidate_kind(code, args, loc)[1] for code in selected_codes),
            background_representatives=[code.hex() for code in representatives],
            original_normalization=[minimum, maximum], linear=linear, native_shift=native_shift,
            native_pair=native_pair, recovery_shift=shifts, recovery_pair=recovery_pair, scores=scores,
            all_in_native_pool=all(code in indexed for code in pool),
            original_union_indices=list(self.current.get('selected', [])),
            prefix_audits=audits, common_incumbent=adoption,
            depth_cap_after_common=int(solver_frame().f_locals['depth_cap']),
            state_features=encode([feature._asdict() for feature in features]))
        return json.loads(json.dumps(packet, ensure_ascii=False))

    def commit(self, selected_codes, args):
        if self.cut is None:
            frame = solver_frame()
            loc = frame.f_locals
            choice = self.eligible(selected_codes, args, loc)
            if choice is not None:
                self.clock.mark_boundary()
                state, start = root_boundary_state(args)
                contract_signature = fingerprint(self.contract.stats)
                if self.expected is not None:
                    mismatch = [key for key in sorted(set(state) | set(self.expected['state']))
                                if state.get(key) != self.expected['state'].get(key)]
                    if mismatch:
                        raise RuntimeError('Root prefix mismatch: ' + json.dumps(mismatch[:20]))
                    if contract_signature != self.expected['contract_stats_signature']:
                        raise RuntimeError('Canonical contract prefix mismatch')
                self.clock.continue_live()
                began = REAL_TIME()
                self.packet = self.build_packet(selected_codes, args, loc, choice)
                packet_signature = fingerprint(self.packet)
                if self.expected is not None and packet_signature != self.expected['packet_signature']:
                    raise RuntimeError('Shared pre-choice packet drift')
                self.cut = dict(layer=loc['layer_index'], state=state, state_signature=fingerprint(state),
                    contract_stats_signature=contract_signature, cutoff=self.clock.cutoff,
                    clock_calls=len(self.clock.values), prefix_sec=self.clock.cutoff - start,
                    packet_signature=packet_signature, packet_sec=REAL_TIME() - began)
                if self.expected is None:
                    raise CaptureReady()
                original = list(selected_codes)
                pool = [bytes.fromhex(row['code']) for row in self.packet['pool']]
                positions = self.packet['positions']
                replacement = render_window(original, positions, pool, PAIRS[self.pair], choice[2])
                if sum(candidate_kind(code, args, loc)[0] for code in replacement) != self.packet['macro_count']:
                    raise RuntimeError('Macro count changed')
                if sum(candidate_kind(code, args, loc)[1] for code in replacement) != self.packet['edge_count']:
                    raise RuntimeError('Actual macro-edge count changed')
                added, removed = set(replacement) - set(original), set(original) - set(replacement)
                loc['selected_by_qubo'].difference_update(pool)
                sources = getattr(args, '_frontier_trace_last_selected_sources', {})
                for code in removed:
                    sources.pop(code, None)
                for position in positions:
                    sources[replacement[position]] = 'research_value_slot'
                for local_name, arg_name in (('macro_chain_witness_codes', '_macro_chain_witness_selected_total'),
                    ('unified_macro_supply_codes', '_unified_macro_supply_selected_total')):
                    tracked = loc.get(local_name, set())
                    if hasattr(args, arg_name):
                        setattr(args, arg_name, getattr(args, arg_name) + len(added & tracked) - len(removed & tracked))
                selected_codes[:] = replacement
                self.expected_frontier = [(code, loc['candidates'][code][0], loc['candidates'][code][1]) for code in replacement]
                self.decision = dict(pair=self.pair, positions=positions, added=[c.hex() for c in sorted(added)],
                    removed=[c.hex() for c in sorted(removed)], width=len(replacement),
                    macro_count=self.packet['macro_count'], edge_count=self.packet['edge_count'],
                    scores=self.packet['scores'][self.pair], common_incumbent=self.packet['common_incumbent'])
                # The suffix is governed by work, with local timers advancing on
                # a deterministic virtual clock. Keep the global deadline beyond
                # both declared horizons until the explicit work guard expires.
                frame.f_locals['deadline'] = self.clock.cutoff + 100.
                args._overall_search_deadline = self.clock.cutoff + 100.
                if frame.f_locals['deadline'] != args._overall_search_deadline:
                    raise RuntimeError('Suffix deadline did not update live frame')
                self.originals['_select_next_frontier'] = beam._select_next_frontier
                self.originals['CompactBeamResult'] = beam.CompactBeamResult
                beam._select_next_frontier, beam.CompactBeamResult = self.check_next, self.result
                beam._record_quality_reservoir_selected_telemetry = self.originals['_record_quality_reservoir_selected_telemetry']
                self.budget = ExpansionBudget(self.clock, args, self.horizon, pool, self.maybe_verify)
                self.budget.install()
        return self.originals['_record_quality_reservoir_selected_telemetry'](selected_codes, args)

    def maybe_verify(self):
        if not self.frontier_verified:
            loc = solver_frame().f_locals
            if loc['layer_index'] > self.cut['layer']:
                self.verify()

    def restore(self):
        if self.budget is not None:
            self.budget.restore()
        super().restore()
