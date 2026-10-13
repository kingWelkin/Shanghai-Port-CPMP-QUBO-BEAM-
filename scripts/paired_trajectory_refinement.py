"""Paid, interleaved refinement of independent complete-path trajectories."""
from __future__ import annotations

from dataclasses import asdict

from container_reshuffle.core import Move
from production_selector_replay import REAL_TIME,fingerprint
from single_seed_conditioned_refinement import run as pass_run
from cyclic_conditioned_refinement import run as cycle_run

MODES=('single_cycle','pair_cycle','qubo_cycle','single_pair','qubo_pair','oracle_pair','gated_qubo_pair')
SOLO=dict(single_cycle='single',pair_cycle='pair',qubo_cycle='single_qubo')
PRIMARY=dict(single_pair='single',qubo_pair='single_qubo',oracle_pair='single_oracle',
             gated_qubo_pair='single_qubo')


def run(instance,path,mode,start,budget,publish):
    if mode not in MODES:raise ValueError(mode)
    if mode in SOLO:
        return cycle_run(instance,path,SOLO[mode],start,budget,publish)
    remaining=start+budget-REAL_TIME()
    if mode=='gated_qubo_pair' and remaining<2.:
        publish(dict(type='PORTFOLIO_GATE',remaining_sec=remaining,threshold_sec=2.,
                     selected_policy='single_cycle',absolute_start=start,budget_sec=budget))
        result=cycle_run(instance,path,'single',start,budget,publish)
        return dict(result,mode=mode,gate='single_cycle_below_2s',used_qubo=False)
    policies=(PRIMARY[mode],'pair')
    trajectories=[tuple(path),tuple(path)]
    active=[True,True];passes=[0,0];global_best=tuple(path);records=[]
    q_search_starts=0;credited_joint_paths=0
    deadline=start+budget
    while any(active) and REAL_TIME()<deadline:
        for branch in (0,1):
            if not active[branch] or REAL_TIME()>=deadline:continue
            before=len(trajectories[branch]);round_index=passes[branch]
            publish(dict(type='TRAJECTORY_BEGIN',branch=branch,round=round_index,policy=policies[branch],
                before=before,global_moves=len(global_best),absolute_start=start,budget_sec=budget,
                path_signature=fingerprint([asdict(move) for move in trajectories[branch]]),
                elapsed_sec=REAL_TIME()-start))
            def emit(event):
                nonlocal global_best,q_search_starts,credited_joint_paths
                event=dict(event,trajectory_branch=branch,trajectory_round=round_index,
                           trajectory_policy=policies[branch])
                if event['type']=='PHASE' and event.get('phase')=='qubo_search':q_search_starts+=1
                if event['type']=='INCUMBENT':
                    if event['moves']>=len(global_best):return
                    candidate=tuple(Move(**value) for value in event['path'])
                    if len(candidate)!=event['moves']:raise ValueError('Wrong complete-path length')
                    ready=REAL_TIME()-start
                    if ready>budget:return
                    # The inner pass has already physically replayed this full
                    # path. The driver independently replays every publication.
                    global_best=candidate
                    credited_joint_paths+=event['source']=='qubo_joint'
                    event.update(ready_sec=ready,source=f'branch{branch}:'+event['source'],
                                 original_source=event['source'])
                elif event['type']=='MODEL':
                    event['record']=dict(event['record'],trajectory_branch=branch,
                        trajectory_round=round_index,trajectory_policy=policies[branch],
                        global_moves=len(global_best))
                publish(event)
            last=pass_run(instance,trajectories[branch],policies[branch],start,budget,emit)
            candidate=tuple(Move(**value) for value in last['path'])
            if len(candidate)>before:raise ValueError('Branch lost its own incumbent')
            trajectories[branch]=candidate;passes[branch]+=1
            active[branch]=bool(candidate and len(candidate)<before)
            record=dict(branch=branch,round=round_index,policy=policies[branch],before=before,
                        after=len(candidate),active=active[branch],global_moves=len(global_best),
                        elapsed_sec=REAL_TIME()-start)
            records.append(record)
            publish(dict(type='TRAJECTORY_END',path=[asdict(move) for move in candidate],**record))
    return dict(mode=mode,moves=len(global_best),path=[asdict(m) for m in global_best],
        seed_kind='explicit_two_trajectory_portfolio',passes=sum(passes),branch_passes=passes,
        branch_moves=list(map(len,trajectories)),branch_active=active,trajectory_records=records,
        backend_end_sec=REAL_TIME()-start,elapsed_sec=REAL_TIME()-start,
        stop_reason='deadline' if REAL_TIME()>=deadline else 'both_trajectories_stalled',
        global_cpmp_optimal=False,configured_primary_policy=PRIMARY[mode],
        used_qubo=bool(q_search_starts),q_search_starts=q_search_starts,
        worker_timely_joint_publications=credited_joint_paths)
