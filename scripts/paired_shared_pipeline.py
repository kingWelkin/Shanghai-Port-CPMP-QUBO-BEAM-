"""Original-clock paired continuation with separate raw and reported ledgers."""
from __future__ import annotations

from dataclasses import asdict
import random
import numpy as np

import wall_budget_cpmp_pipeline as main
from paired_trajectory_refinement import MODES,run as refine
from container_reshuffle.core import Move
from production_selector_replay import REAL_TIME,fingerprint


def run_continuation(instance,mode,handoff,start,budget,publish):
    if mode not in MODES:raise ValueError(mode)
    if handoff['absolute_start']!=start or handoff['budget_sec']!=budget:
        raise ValueError('Continuation clock was rebased')
    path=tuple(Move(**v) for v in handoff['path'])
    if len(path)!=handoff['moves'] or fingerprint(handoff['path'])!=handoff['path_signature']:
        raise ValueError('Common input changed in transit')
    random.seed(37);np.random.seed(37);context=None
    def emit_reported(event):
        if context is not None:event=dict(event,refinement_provenance=dict(context))
        publish(event)
    ledger=main.CompletePathLedger(instance,start,budget,emit_reported)
    ledger.offer(path,'shared_handoff')
    publish(dict(type='BRANCH_INPUT',path_signature=handoff['path_signature'],moves=len(path),
        model_input_signature=None if ledger.best is None else fingerprint([asdict(m) for m in ledger.best]),
        model_input_moves=None if ledger.best is None else len(ledger.best),
        absolute_start=start,budget_sec=budget,elapsed_sec=REAL_TIME()-start))
    if ledger.best is None or REAL_TIME()>=ledger.deadline:
        return dict(mode=mode,skipped=True,reason='NO_REMAINING_TIME',elapsed_sec=REAL_TIME()-start)
    def emit_refinement(event):
        nonlocal context
        if event['type']=='INCUMBENT':
            # This raw, already validated path is telemetry only. The outer
            # ledger charges validation/output cleanup and publishes the result.
            publish(dict(event,type='REFINEMENT_INCUMBENT',stage='refinement'))
            context=dict(trajectory_branch=event.get('trajectory_branch'),
                trajectory_round=event.get('trajectory_round'),trajectory_policy=event.get('trajectory_policy'),
                original_source=event.get('original_source',event['source']),
                raw_source=event['source'],raw_moves=event['moves'])
            try:
                ledger.offer(tuple(Move(**v) for v in event['path']),'refine:'+event['source'])
            finally:context=None
        else:
            publish(dict(event,stage='refinement',reported_global_moves=len(ledger.best)))
    result=refine(instance,ledger.best,mode,start,budget,emit_refinement)
    ledger.offer(tuple(Move(**v) for v in result['path']),'refinement_final')
    return dict(mode=mode,moves=len(ledger.best),elapsed_sec=REAL_TIME()-start,
                ledger=ledger.stats,refinement_passes=result.get('passes'),global_cpmp_optimal=False)
