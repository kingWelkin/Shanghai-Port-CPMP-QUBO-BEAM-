"""Fresh native/common producer and independent timed post-prefix methods."""
from __future__ import annotations

from dataclasses import asdict
import random
import numpy as np

import wall_budget_cpmp_pipeline as main
import bounded_conditioned_refinement as bounded
from fast_conditioned_refinement import run as refine
from container_reshuffle.core import Move
from paper_priority_group_contract import PriorityGroupContract
from research_receiver_allocation import physical_replay
from production_selector_replay import REAL_TIME, fingerprint

MODES = ('frg', 'single', 'pair', 'qubo', 'oracle')


def run_prefix(entry, start, budget, publish, reserve=2.):
    saved = main.capped
    def handoff(instance, path, mode, shared_start, shared_budget, post_publish, **kwargs):
        if shared_start != start or shared_budget != budget:
            raise ValueError('Native producer changed its original clock')
        physical_replay(instance.stacks, instance.max_height, path, goal=True)
        serialized = [asdict(move) for move in path]
        publish(dict(type='HANDOFF', path=serialized, moves=len(path),
            path_signature=fingerprint(serialized), ready_sec=REAL_TIME() - start,
            absolute_start=start, budget_sec=budget))
        return dict(path=serialized)
    try:
        main.capped = handoff
        return main.run(entry, 'pair', start, budget, publish, reserve=reserve)
    finally:
        main.capped = saved


def run_continuation(instance, mode, handoff, start, budget, publish):
    if mode not in MODES:
        raise ValueError(mode)
    if handoff['absolute_start'] != start or handoff['budget_sec'] != budget:
        raise ValueError('Continuation clock was rebased')
    random.seed(37)
    np.random.seed(37)
    path = tuple(Move(**value) for value in handoff['path'])
    if len(path) != handoff['moves'] or fingerprint(handoff['path']) != handoff['path_signature']:
        raise ValueError('Shared native path changed in transit')
    ledger = main.CompletePathLedger(instance, start, budget, publish)
    ledger.offer(path, 'shared_handoff')
    publish(dict(type='BRANCH_INPUT', path_signature=handoff['path_signature'], moves=len(path),
        model_input_signature=None if ledger.best is None else fingerprint([asdict(move) for move in ledger.best]),
        model_input_moves=None if ledger.best is None else len(ledger.best),
        absolute_start=start, budget_sec=budget, elapsed_sec=REAL_TIME() - start))
    if ledger.best is None or REAL_TIME() >= ledger.deadline:
        return dict(mode=mode, skipped=True, reason='NO_REMAINING_TIME', elapsed_sec=REAL_TIME() - start)
    def post_publish(message):
        if message['type'] == 'INCUMBENT':
            ledger.offer(tuple(Move(**value) for value in message['path']), 'refine:' + message['source'])
        else:
            publish(dict(message, stage='refinement'))
    mapped = dict(frg='frg', single='pair_frg', pair='pair_frg', qubo='capped_qubo_frg',
                  oracle='capped_oracle_frg')[mode]
    saved = bounded.classical_seed
    try:
        if mode == 'single':
            bounded.classical_seed = lambda model, pairs=True: saved(model, pairs=False)
        result = refine(instance, ledger.best, mapped, start, budget, post_publish, continue_frg=True)
    finally:
        bounded.classical_seed = saved
    ledger.offer(tuple(Move(**value) for value in result['path']), 'refinement_final')
    return dict(mode=mode, moves=len(ledger.best), elapsed_sec=REAL_TIME() - start,
                ledger=ledger.stats, global_cpmp_optimal=False)
