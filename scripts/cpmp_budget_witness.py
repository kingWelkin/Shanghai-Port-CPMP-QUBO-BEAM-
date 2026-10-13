"""Physically replay cached CPMP recovery witnesses; never search for new paths."""
from __future__ import annotations
from dataclasses import asdict,replace
import benchmark_compact_beam_qubo_n0 as beam
from production_selector_replay import fingerprint


def normalized_steps(moves):
    return [replace(move,step=i) for i,move in enumerate(moves)]


def replay_prefix(instance,code,data,parent,macro_paths,macro_parents):
    record=dict(code=code.hex(),recorded_depth=int(data[1]),recorded_n=int(data[0]),
        prefix_verified=False,actual_prefix_cost=None,prefix_status='UNKNOWN_PREFIX')
    if beam._parent_chain_would_cycle(parent,code,data[2]):
        record['prefix_status']='PARENT_CYCLE';return record,None,None
    local_parent=dict(parent);local_parent[code]=data[2]
    try:
        # Expand ancestry; actual lifting/replay validates every edge below.
        codes=beam._backtrack_with_macro_edges(local_parent,code,macro_paths,macro_parent_by_code=macro_parents)
        moves=normalized_steps(beam._codes_to_moves_instance(instance,codes))
        state=beam.create_initial_state(instance)
        for move in moves: state=beam.apply_move(instance,state,move)
        endpoint,n=beam._encode(state.stacks,instance.stack_count,instance.max_height)
    except (KeyError,ValueError,RuntimeError,IndexError) as error:
        record['prefix_status']='UNLIFTABLE_PREFIX';record['prefix_error']=str(error)
        return record,None,None
    if endpoint!=code or n!=int(data[0]):
        record['prefix_status']='ENDPOINT_MISMATCH';return record,None,None
    record.update(prefix_verified=True,actual_prefix_cost=len(moves),prefix_status='VALID_PREFIX',
        prefix_moves=[asdict(m) for m in moves],prefix_signature=fingerprint([asdict(m) for m in moves]),
        depth_difference=len(moves)-int(data[1]))
    return record,moves,state


def audit_cached_route(instance,code,data,parent,macro_paths,macro_parents,rollout,cap):
    record,prefix,state=replay_prefix(instance,code,data,parent,macro_paths,macro_parents)
    record.update(cached=rollout is not None,claimed_complete=bool(rollout is not None and rollout.feasible and rollout.final_n==0),
        complete_status='UNKNOWN_NO_CACHED',business_cap=int(cap),full_moves=None,full_cost=None,
        goal_ok=None,budget_ok=None,max_g_n=None)
    if rollout is None: return record,None
    record.update(recorded_rollout_moves=int(rollout.moves),recorded_rollout_status=rollout.status,
        recorded_rollout_n=int(rollout.final_n))
    if prefix is None:
        record['complete_status']='UNVERIFIED_PREFIX';return record,None
    if not rollout.move_path and int(data[0])!=0:
        record['complete_status']='UNVERIFIED_NO_TAIL';return record,None
    stack_map=beam._canonical_to_physical_stack_map(
        beam._code_to_stacks(code,instance.stack_count,instance.max_height),state.stacks)
    if stack_map is None:
        record['complete_status']='INVALID_STACK_MAP';return record,None
    translated=beam._translate_moves_by_stack_map(rollout.move_path,stack_map)
    if translated is None:
        record['complete_status']='INVALID_TAIL_MAP';return record,None
    moves=normalized_steps(prefix+translated)
    solution=beam.Solution(solver_name='cached-witness-audit',feasible=True,optimal=False,
        objective=float(len(moves)),runtime_sec=0.0,moves=moves,status='AUDITED')
    try:
        goal_ok,message,_=beam.validate_solution(instance,solution)
    except (ValueError,RuntimeError,IndexError) as error:
        goal_ok=False;message=str(error)
    record.update(goal_ok=goal_ok,goal_message=message,full_cost=len(moves),
        full_moves=[asdict(m) for m in moves],full_signature=fingerprint([asdict(m) for m in moves]))
    if not goal_ok:
        record['complete_status']='INVALID_COMPLETE_CLAIM' if record['claimed_complete'] else 'PARTIAL_OR_INVALID_PATH'
        return record,None
    peak=beam.max_entropy_along_moves(instance,moves)
    record.update(max_g_n=int(peak),budget_ok=peak<=cap,
        complete_status='VALID_BUDGET' if peak<=cap else 'VALID_OVER_BUDGET')
    return record,moves if peak<=cap else None


def adopt_common_witness(frame,certified):
    """Adopt a strictly better validated cached path through Python 3.13 locals."""
    loc=frame.f_locals
    costs=[]
    if loc['rollout_terminal_moves'] is not None: costs.append(len(loc['rollout_terminal_moves']))
    if loc.get('frg_seed') is not None and loc.get('frg_seed_accepted',False): costs.append(len(loc['frg_seed'].moves))
    if loc.get('wang_incumbent') is not None: costs.append(len(loc['wang_incumbent'].moves))
    old=min(costs) if costs else None
    if not certified: return dict(adopted=False,old_best_cost=old,new_cost=None,source_code=None)
    code,moves=min(certified,key=lambda item:(len(item[1]),item[0]))
    if old is not None and len(moves)>=old:
        return dict(adopted=False,old_best_cost=old,new_cost=len(moves),source_code=code.hex())
    loc['rollout_terminal_moves']=list(moves)
    loc['rollout_terminal_source']='audited_cached_witness'
    loc['depth_cap']=min(int(loc['depth_cap']),max(0,len(moves)-1))
    if frame.f_locals['rollout_terminal_moves']!=moves: raise RuntimeError('Shared incumbent did not update live solver locals')
    return dict(adopted=True,old_best_cost=old,new_cost=len(moves),source_code=code.hex(),
        depth_cap=int(frame.f_locals['depth_cap']),path_signature=fingerprint([asdict(m) for m in moves]))
