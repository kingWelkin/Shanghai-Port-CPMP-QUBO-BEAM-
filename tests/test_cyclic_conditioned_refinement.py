from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cyclic_conditioned_refinement as cyclic
from test_bounded_conditioned_refinement import example


def test_real_cycles_keep_original_deadline_and_own_monotone_path(monkeypatch):
    instance,path,_=example();saved=cyclic.pass_run;calls=[]
    start=time.perf_counter()
    def observed(instance,current,mode,original_start,budget,publish,**kwargs):
        calls.append((len(current),original_start,budget))
        return saved(instance,current,mode,original_start,budget,publish,**kwargs)
    monkeypatch.setattr(cyclic,'pass_run',observed)
    events=[];result=cyclic.run(instance,path,'single_qubo',start,2.,events.append)
    assert result['moves']==2 and result['stop_reason']=='strict_cycle_stalled'
    assert result['passes']==2 and [r[0] for r in calls]==[3,2]
    assert all(r[1]==start and r[2]==2. for r in calls)
    publications=[e['moves'] for e in events if e['type']=='INCUMBENT']
    assert publications==sorted(set(publications),reverse=True)
    assert result['global_cpmp_optimal'] is False


def test_one_pass_comparison_runs_only_one_original_pass(monkeypatch):
    instance,path,_=example();saved=cyclic.pass_run;calls=[]
    def observed(*args,**kwargs):
        calls.append(args[2]);return saved(*args,**kwargs)
    monkeypatch.setattr(cyclic,'pass_run',observed)
    result=cyclic.run(instance,path,'single_qubo_once',time.perf_counter(),2.,lambda e:None)
    assert result['moves']==2 and calls==['single_qubo']
