from dataclasses import asdict
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from paired_shared_pipeline import run_continuation
from production_selector_replay import fingerprint
from container_reshuffle.core import Move
from research_receiver_allocation import physical_replay
from test_bounded_conditioned_refinement import example


def test_paired_continuation_keeps_clock_and_separates_raw_from_reported_paths():
    instance,path,_=example();start=time.perf_counter();values=[asdict(m) for m in path]
    handoff=dict(path=values,moves=len(path),path_signature=fingerprint(values),absolute_start=start,budget_sec=2.)
    events=[];result=run_continuation(instance,'qubo_pair',handoff,start,2.,events.append)
    assert result['moves']==2
    assert next(e for e in events if e['type']=='BRANCH_INPUT')['absolute_start']==start
    raw=[e for e in events if e['type']=='REFINEMENT_INCUMBENT']
    reported=[e for e in events if e['type']=='INCUMBENT']
    assert raw and reported[-1]['moves']==2 and reported[-1]['refinement_provenance']['trajectory_branch']==0
    for event in events:
        if event['type'] in ('INCUMBENT','REFINEMENT_INCUMBENT','TRAJECTORY_END'):
            physical_replay(instance.stacks,instance.max_height,tuple(Move(**v) for v in event['path']),goal=True)
        if event['type']=='TRAJECTORY_BEGIN':assert event['absolute_start']==start and event['budget_sec']==2.
    with pytest.raises(ValueError,match='rebased'):
        run_continuation(instance,'qubo_pair',handoff,start+1.,2.,events.append)
