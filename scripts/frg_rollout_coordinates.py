"""Opt-in canonical child-start rollout writers and physically checked grafts."""
from __future__ import annotations
import ast
from dataclasses import replace,asdict
import inspect
import benchmark_compact_beam_qubo_n0 as beam
from production_selector_replay import fingerprint


HELPER='_research_canonical_macro_tail'


def canonical_rollout(reference_stacks,max_height,rollout):
    """Permute a reference-layout tail into canonical child coordinates."""
    stacks=[list(s) for s in reference_stacks]
    code,n=beam._encode(stacks,len(stacks),max_height)
    canonical=beam._code_to_stacks(code,len(stacks),max_height)
    mapping=beam._canonical_to_physical_stack_map(canonical,stacks)
    if mapping is None or len(mapping)!=len(stacks): raise ValueError('Reference layout is not a stack permutation')
    inverse={layout:logical for logical,layout in mapping.items()}
    moves=[]
    for i,m in enumerate(rollout.move_path):
        if m.from_stack not in inverse or m.to_stack not in inverse: raise ValueError('Move index outside reference layout')
        moves.append(replace(m,step=i,from_stack=inverse[m.from_stack],to_stack=inverse[m.to_stack]))
    result=replace(rollout,moves=len(moves),move_path=tuple(moves))
    return result,dict(code=code.hex(),reference_to_canonical=[inverse[i] for i in range(len(stacks))],
        coordinate_changed=any(inverse[i]!=i for i in range(len(stacks))),
        moves_changed=tuple(moves)!=rollout.move_path)


def move_core(move):
    return (move.box,move.from_stack,move.to_stack,move.from_level,move.to_level)


def canonical_macro_tail(child_layout,max_height,prefix_moves,witness_moves,feasible,final_n,status,runtime_sec):
    prefix=tuple(prefix_moves);full=tuple(witness_moves)
    if len(full)<len(prefix) or [move_core(m) for m in full[:len(prefix)]]!=[move_core(m) for m in prefix]:
        raise ValueError('Macro witness does not start with its executed prefix')
    tail=full[len(prefix):]
    raw=beam.UnionRolloutProbeResult(bool(feasible),int(final_n),len(tail),str(status),float(runtime_sec),tuple(tail))
    return canonical_rollout(child_layout.stacks,max_height,raw)[0]


def macro_recorder_with_correct_tail(original):
    tree=ast.parse(inspect.getsource(original));replaced=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='new_rollout' for t in node.targets):
            if not isinstance(node.value,ast.Call) or not isinstance(node.value.func,ast.Name) or node.value.func.id!='UnionRolloutProbeResult':
                raise ValueError('Native macro writer changed its rollout constructor')
            node.value=ast.Call(func=ast.Name(id=HELPER,ctx=ast.Load()),args=[ast.Name(id=n,ctx=ast.Load()) for n in (
                'child_layout','max_height','prefix_moves','witness_moves','rollout_feasible','rollout_final_n','rollout_status','rollout_runtime_sec')],keywords=[])
            replaced+=1
    if replaced!=1: raise ValueError('Expected exactly one macro rollout construction')
    ast.fix_missing_locations(tree)
    namespace=dict(beam.__dict__)
    namespace[HELPER]=canonical_macro_tail
    exec(compile(tree,original.__code__.co_filename+'<canonical-tail>','exec'),namespace)
    return namespace[original.__name__]


def replay_reference(stacks,max_height,moves):
    instance=beam.Instance('rollout-reference',beam.ProblemVariant.PREMARSHALLING_CPMP,len(stacks),max_height,
        sum(len(s) for s in stacks),[list(s) for s in stacks])
    state=beam.create_initial_state(instance)
    for i,m in enumerate(moves):
        # Check stored level annotations too; do not repair a bad physical path.
        if m.from_level is not None and m.from_level!=len(state.stacks[m.from_stack]): raise ValueError('Reference from-level mismatch')
        if m.to_level is not None and m.to_level!=len(state.stacks[m.to_stack])+1: raise ValueError('Reference to-level mismatch')
        state=beam.apply_move(instance,state,replace(m,step=i))
    code,n=beam._encode(state.stacks,instance.stack_count,max_height)
    return state.stacks,code,n


def graft_child_tail(code,data,parent,instance,rollout,cap,macro_paths=None,macro_parents=None,respect_cap=True):
    if not rollout.feasible or rollout.final_n!=0: return None
    if beam._parent_chain_would_cycle(parent,code,data[2]): return None
    local_parent=dict(parent);local_parent[code]=data[2]
    try:
        codes=beam._backtrack_with_macro_edges(local_parent,code,macro_paths or {},macro_parent_by_code=macro_parents)
        prefix=beam._codes_to_moves_instance(instance,codes)
        state=beam.create_initial_state(instance)
        for m in prefix: state=beam.apply_move(instance,state,m)
        mapping=beam._canonical_to_physical_stack_map(beam._code_to_stacks(code,instance.stack_count,instance.max_height),state.stacks)
        if mapping is None: return None
        tail=beam._translate_moves_by_stack_map(rollout.move_path,mapping)
        if tail is None: return None
        moves=[replace(m,step=i) for i,m in enumerate(prefix+tail)]
        solution=beam.Solution(solver_name='canonical-cached-tail',feasible=True,optimal=False,objective=float(len(moves)),
            runtime_sec=0.0,moves=moves,status='GRAFT_CHECK')
        ok,message,_=beam.validate_solution(instance,solution)
        if not ok or (respect_cap and beam.max_entropy_along_moves(instance,moves)>cap): return None
        return moves
    except (KeyError,ValueError,RuntimeError,IndexError): return None


class CanonicalTailContract:
    """Scoped opt-in repair; original source and default bindings remain intact."""
    def __init__(self):
        self.originals={};self.records=[]
        self.stats=dict(frontier_calls=0,frontier_conversions=0,macro_record_calls=0,macro_records=0,
            macro_prefix_moves_removed=0,ordinary_calls=0,ordinary_steps_normalized=0,graft_calls=0,graft_accepted=0)

    def install(self):
        if self.originals: raise RuntimeError('Contract already installed')
        self.originals={name:getattr(beam,name) for name in ('_frg_frontier_child_certification_rollout',
            '_record_macro_chain_witness_candidate','_union_rollout_probe_completion','_completion_witness_grafted_moves')}
        self.macro_impl=macro_recorder_with_correct_tail(self.originals['_record_macro_chain_witness_candidate'])
        beam._frg_frontier_child_certification_rollout=self.frontier
        beam._record_macro_chain_witness_candidate=self.macro
        beam._union_rollout_probe_completion=self.ordinary
        beam._completion_witness_grafted_moves=self.graft

    def restore(self):
        for name,original in self.originals.items(): setattr(beam,name,original)
        self.originals.clear()

    def frontier(self,child_layout,max_height,start_step,args):
        self.stats['frontier_calls']+=1
        raw=self.originals['_frg_frontier_child_certification_rollout'](child_layout,max_height,start_step,args)
        if raw is None: return None
        result,info=canonical_rollout(child_layout.stacks,max_height,raw)
        self.stats['frontier_conversions']+=info['coordinate_changed']
        if len(self.records)<200: self.records.append(dict(kind='frontier',**info,tail_moves=result.moves))
        return result

    def ordinary(self,code,data,args):
        self.stats['ordinary_calls']+=1
        raw=self.originals['_union_rollout_probe_completion'](code,data,args)
        canonical=beam._code_to_stacks(code,args._stack_count,args._max_height)
        result,info=canonical_rollout(canonical,args._max_height,raw)
        self.stats['ordinary_steps_normalized']+=info['moves_changed']
        return result

    def macro(self,*args,**kwargs):
        self.stats['macro_record_calls']+=1
        code=self.macro_impl(*args,**kwargs)
        if code is not None:
            self.stats['macro_records']+=1
            self.stats['macro_prefix_moves_removed']+=len(kwargs['prefix_moves'])
            if len(self.records)<200:
                self.records.append(dict(kind='macro',code=code.hex(),prefix_removed=len(kwargs['prefix_moves']),
                    tail_moves=(kwargs.get('args') or args[1])._union_rollout_probe_cache[code].moves))
        return code

    def graft(self,code,data,parent,instance,rollout,*,cap,macro_paths=None,respect_cap=True):
        self.stats['graft_calls']+=1
        macro_parents=None
        frame=inspect.currentframe().f_back
        while frame is not None:
            config=frame.f_locals.get('args')
            if config is not None and hasattr(config,'_frg_macro_edge_parent_by_code'):
                macro_parents=config._frg_macro_edge_parent_by_code;break
            frame=frame.f_back
        moves=graft_child_tail(code,data,parent,instance,rollout,cap,macro_paths,macro_parents,respect_cap)
        self.stats['graft_accepted']+=moves is not None
        return moves

    def migrate_current(self,args,candidates,rollout_by_code=None):
        """Replay all legacy entries; repair proven current origins or invalidate."""
        cache=getattr(args,'_union_rollout_probe_cache',{})
        parents=getattr(args,'_frg_macro_edge_parent_by_code',{})
        prefixes=getattr(args,'_frg_macro_edge_moves_by_code',{})
        current_frontier=set(getattr(args,'_frg_frontier_child_codes',set()))
        current_chain={c for c in candidates if beam._union_diversity_candidate_origin(c,args) in ('frg_macro_chain_witness','unified_macro_supply')}
        current=set(cache)
        audit=[]
        for code in sorted(current):
            raw=cache.get(code);parent_code=parents.get(code);prefix=prefixes.get(code)
            row=dict(code=code.hex(),status='SKIPPED_UNVERIFIED_SOURCE')
            if raw is None: audit.append(row);continue
            origin=beam._union_diversity_candidate_origin(code,args)
            try:
                source=raw
                if code in current_frontier or code in current_chain:
                    if parent_code is None or not prefix: raise ValueError('Current macro cache lacks source prefix')
                    parent_stacks=beam._code_to_stacks(parent_code,args._stack_count,args._max_height)
                    reference,endpoint,n=replay_reference(parent_stacks,args._max_height,prefix)
                    if endpoint!=code: raise ValueError('Macro source prefix does not reach cached child')
                    if code in current_chain:
                        if [move_core(m) for m in raw.move_path[:len(prefix)]]!=[move_core(m) for m in prefix]:
                            raise ValueError('Macro cache lacks its advertised parent prefix')
                        source=replace(raw,moves=len(raw.move_path)-len(prefix),move_path=raw.move_path[len(prefix):])
                    basis='verified_macro_reference'
                else:
                    reference=beam._code_to_stacks(code,args._stack_count,args._max_height)
                    basis='canonical_replay_verified'
                _,tail_endpoint,tail_n=replay_reference(reference,args._max_height,source.move_path)
                if tail_n!=raw.final_n: raise ValueError('Cached tail endpoint does not match metadata')
                if raw.feasible and tail_n!=0: raise ValueError('Feasible cache entry does not reach the goal')
                result,info=canonical_rollout(reference,args._max_height,source)
                _,check_endpoint,check_n=replay_reference(beam._code_to_stacks(code,args._stack_count,args._max_height),args._max_height,result.move_path)
                if check_endpoint!=tail_endpoint or check_n!=tail_n: raise ValueError('Canonical tail changed its result')
                cache[code]=result
                views=[getattr(args,'_union_rollout_probe_current_results',{}),rollout_by_code or {}]
                for view in views:
                    if code in view: view[code]=result
                row.update(status='CANONICAL_CHILD_TAIL',origin=origin,source_basis=basis,**info,tail_moves=result.moves,
                    prefix_removed=raw.moves-result.moves,original_tail_signature=fingerprint([asdict(m) for m in raw.move_path]),
                    canonical_tail_signature=fingerprint([asdict(m) for m in result.move_path]))
            except (KeyError,ValueError,RuntimeError,IndexError) as error:
                row.update(status='INVALIDATED_UNVERIFIED',reason=str(error))
                cache.pop(code,None)
                for view in [getattr(args,'_union_rollout_probe_current_results',{}),rollout_by_code or {}]: view.pop(code,None)
            audit.append(row)
        return audit
