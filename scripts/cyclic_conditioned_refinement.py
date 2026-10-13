"""Repeat paid route/FRG passes on actual incumbents under the original clock."""
from __future__ import annotations

from dataclasses import asdict

from container_reshuffle.core import Move
from production_selector_replay import REAL_TIME
from single_seed_conditioned_refinement import run as pass_run

MODES=('single','pair','single_qubo','single_oracle',
       'single_once','pair_once','single_qubo_once','pair_qubo')
ONCE=dict(single_once='single',pair_once='pair',single_qubo_once='single_qubo',pair_qubo='pair_qubo')


def run(instance,path,mode,start,budget,publish,**kwargs):
    if mode not in MODES:raise ValueError(mode)
    if mode in ONCE:
        return pass_run(instance,path,ONCE[mode],start,budget,publish,**kwargs)
    if kwargs.get('releases') is not None:
        raise ValueError('Each cycle must rebuild its neighborhoods from its actual path')
    current=tuple(path);cycles=[];last=None;deadline=start+budget;reason='deadline'
    while REAL_TIME()<deadline and current:
        cycle=len(cycles);before=len(current)
        publish(dict(type='CYCLE_BEGIN',cycle=cycle,before=before,absolute_start=start,
                     budget_sec=budget,elapsed_sec=REAL_TIME()-start))
        def emit(event):
            event=dict(event,cycle=cycle)
            if event['type']=='MODEL':event['record']=dict(event['record'],cycle=cycle)
            publish(event)
        last=pass_run(instance,current,mode,start,budget,emit,**kwargs)
        candidate=tuple(Move(**value) for value in last['path'])
        if len(candidate)>before:raise ValueError('Cycle lost its physically validated incumbent')
        current=candidate
        record=dict(cycle=cycle,before=before,after=len(current),elapsed_sec=REAL_TIME()-start,
                    complete_with_time_remaining=REAL_TIME()<deadline)
        cycles.append(record);publish(dict(type='CYCLE_END',**record))
        if REAL_TIME()>=deadline:break
        if len(current)==before:
            reason='strict_cycle_stalled';break
    if not current:reason='zero_moves'
    if last is None:
        return dict(mode=mode,path=[asdict(m) for m in current],moves=len(current),models=[],
            seed_kind=None,backend_end_sec=REAL_TIME()-start,elapsed_sec=REAL_TIME()-start,
            passes=0,cycles=[],stop_reason=reason,global_cpmp_optimal=False)
    return dict(last,mode=mode,path=[asdict(m) for m in current],moves=len(current),
                passes=len(cycles),cycles=cycles,stop_reason=reason,elapsed_sec=REAL_TIME()-start,
                global_cpmp_optimal=False)
