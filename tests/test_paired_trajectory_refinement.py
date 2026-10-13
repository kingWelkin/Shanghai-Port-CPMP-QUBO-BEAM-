from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import paired_trajectory_refinement as paired
import sequential_conditioned_refiner as classical
from container_reshuffle.core import Move
from research_receiver_allocation import physical_replay
from test_bounded_conditioned_refinement import example


def test_independent_actual_branches_share_clock_and_publish_global_improvements(monkeypatch):
    instance,path,_=example();start=time.perf_counter();saved=paired.pass_run;calls=[]
    def observed(instance,current,mode,original_start,budget,publish):
        calls.append((mode,len(current),original_start,budget))
        return saved(instance,current,mode,original_start,budget,publish)
    monkeypatch.setattr(paired,'pass_run',observed)
    events=[];result=paired.run(instance,path,'qubo_pair',start,2.,events.append)
    assert result['moves']==2 and result['branch_moves']==[2,2]
    assert [r[:2] for r in calls]==[('single_qubo',3),('pair',3),('single_qubo',2),('pair',2)]
    assert all(r[2]==start and r[3]==2. for r in calls)
    publications=[e for e in events if e['type']=='INCUMBENT']
    assert [e['moves'] for e in publications]==[2]
    first_entries=[e for e in events if e['type']=='TRAJECTORY_BEGIN' and e['round']==0]
    assert len(first_entries)==2 and first_entries[0]['path_signature']==first_entries[1]['path_signature']
    for event in events:
        if event['type']=='TRAJECTORY_END':
            physical_replay(instance.stacks,instance.max_height,tuple(Move(**v) for v in event['path']),goal=True)
    assert result['stop_reason']=='both_trajectories_stalled'
    assert all(e['record']['seed_kind']==('three_order_single' if e['record']['trajectory_branch']==0 else 'four_start_pair')
               for e in events if e['type']=='MODEL')


def test_late_gate_is_explicit_classical_fallback_without_pair_or_q(monkeypatch):
    instance,path,_=example();events=[]
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected paired/Q execution below the gate')
    monkeypatch.setattr(paired,'pass_run',forbidden)
    monkeypatch.setattr(classical,'exchange',forbidden)
    result=paired.run(instance,path,'gated_qubo_pair',time.perf_counter(),1.,events.append)
    assert result['moves']==2 and result['used_qubo'] is False
    assert events[0]['type']=='PORTFOLIO_GATE' and events[0]['selected_policy']=='single_cycle'
    assert not any(e.get('source')=='qubo_joint' for e in events)
