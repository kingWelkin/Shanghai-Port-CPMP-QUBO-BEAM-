from pathlib import Path
import os
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from container_reshuffle.core import Instance,Move,ProblemVariant
import bounded_conditioned_refinement as bounded
from deadline_worker_protocol import run_worker


def example():
    instance=Instance('timed-test',ProblemVariant.PREMARSHALLING_CPMP,4,4,7,[[6,1,3,4],[2],[5],[7]])
    path=(Move(0,4,0,1,4,2),Move(1,3,0,3,3,2),Move(2,4,1,2,2,2))
    return instance,path,[dict(seed=3,boxes=(2,3)),dict(seed=2,boxes=(1,2,3))]


def test_budgeted_controller_publishes_valid_pair_seed_before_joint(monkeypatch):
    instance,path,releases=example()
    for mode in ('pair_frg','qubo_frg','oracle_frg'):
        published=[];original=bounded.exact.milp
        result=bounded.run(instance,path,mode,time.perf_counter(),20.,published.append,
            releases=releases,continue_frg=False)
        assert result['moves']==2 and bounded.exact.milp is original
        candidates=[e for e in published if e['type']=='INCUMBENT']
        assert candidates and candidates[0]['source']=='pair_seed'
        assert all(e['ready_sec']<20. for e in candidates)
        for e in candidates:
            bounded.physical_replay(instance.stacks,4,tuple(Move(**v) for v in e['path']),goal=True)
        assert all(m['nonoracle_milp_calls']==0 for m in result['models'])


def test_expired_publication_keeps_original():
    instance,path,releases=example();published=[]
    result=bounded.run(instance,path,'qubo_frg',time.perf_counter()-5.,.1,published.append,
        releases=releases,continue_frg=False)
    assert result['moves']==3 and not published


def test_parent_hard_deadline_preserves_received_incumbent_and_kills_overrun():
    program=('import json,sys,time\n'
        'print(json.dumps(dict(type="READY")),flush=True)\n'
        'start=json.loads(sys.stdin.readline())\n'
        'print(json.dumps(dict(type="INCUMBENT",moves=4)),flush=True)\n'
        'time.sleep(2)\n'
        'print(json.dumps(dict(type="INCUMBENT",moves=1)),flush=True)\n')
    r=run_worker([sys.executable,'-u','-c',program],cwd=Path.cwd(),env=os.environ.copy(),budget=.3)
    assert r['status']=='TIME_LIMIT' and r['termination_lag_sec']<.15
    assert [v['moves'] for v in r['messages'] if v['type']=='INCUMBENT' and v['received_sec']<=.3]==[4]


def test_parent_reports_early_completion():
    program=('import json,sys\nprint(json.dumps(dict(type="READY")),flush=True)\n'
        'sys.stdin.readline()\nprint(json.dumps(dict(type="DONE")),flush=True)\n')
    r=run_worker([sys.executable,'-u','-c',program],cwd=Path.cwd(),env=os.environ.copy(),budget=.5)
    assert r['status']=='COMPLETE' and r['stop_requested_sec']<.5
