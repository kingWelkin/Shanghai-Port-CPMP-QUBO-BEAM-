"""Compare actually computed single and Pair seeds for the same route QUBO."""
from __future__ import annotations

import bounded_conditioned_refinement as original
from fast_conditioned_refinement import run as fast_run

MODES = ('single', 'single_qubo', 'single_oracle', 'pair', 'pair_qubo', 'frg')
MAPPED = dict(single='pair_frg', single_qubo='capped_qubo_frg',
              single_oracle='capped_oracle_frg', pair='pair_frg',
              pair_qubo='capped_qubo_frg', frg='frg')


def run(instance, path, mode, start, budget, publish, **kwargs):
    if mode not in MODES:
        raise ValueError(mode)
    cheap = mode.startswith('single')
    kind = 'three_order_single' if cheap else 'four_start_pair'
    saved = original.classical_seed

    def seed(model, pairs=True):
        chosen, metadata = saved(model, pairs=not cheap)
        metadata = dict(metadata, seed_kind=kind,
                        actual_seed_selection=list(map(int, chosen)))
        return chosen, metadata

    def emit(message):
        message = dict(message, refinement_method=mode)
        if message['type'] == 'PHASE' and message.get('phase') == 'pair_seed':
            message['seed_kind'] = kind
            if cheap:
                message['phase'] = 'single_seed'
        if message['type'] == 'INCUMBENT' and message.get('source') == 'pair_seed' and cheap:
            message['source'] = 'single_seed'
        if message['type'] == 'MODEL':
            record = dict(message['record'], seed_kind=kind)
            if record['status'] == 'SUPPORTED':
                chosen = record['seed_metadata']['actual_seed_selection']
                record['changed_groups_from_seed'] = sum(
                    a != b for a, b in zip(chosen, record['selected'], strict=True))
            message['record'] = record
        publish(message)

    try:
        original.classical_seed = seed
        result = fast_run(instance, path, MAPPED[mode], start, budget, emit, **kwargs)
        return dict(result, mode=mode, seed_kind=None if mode == 'frg' else kind)
    finally:
        original.classical_seed = saved
