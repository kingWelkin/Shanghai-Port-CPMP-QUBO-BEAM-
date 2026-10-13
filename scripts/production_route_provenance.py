"""Fixed-macro-count membership interventions with native suffix provenance."""
from __future__ import annotations
import argparse
from array import array
import gzip
import inspect
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'scripts',ROOT/'src'):
    sys.path.insert(0,str(path))
import numpy as np
import benchmark_compact_beam_qubo_n0 as beam
from beam_selector_research import SelectorObserver
from probe_business_macro_windows import business_args,load_instance,manifest as business_manifest
from production_selector_replay import REAL_TIME,boundary_state,fingerprint,REPLAY
from production_selector_factorial import SuffixClock,WorkBudget,WORK_SETTINGS,SUCCESSOR_LIMIT
from production_route_membership import manifest as round7_manifest,OUT as ROUND7
from run_beam_selector_research import ENVIRONMENT,digest,write_csv,write_json

OUT=ROOT/'docs/experiments/beam_selector_round8_20261002'
CASE='C2_26_single'
ALTERNATIVES=(18,20,21,22,24)
CELLS={**{f'A{m}':(23,19,m,16) for m in ALTERNATIVES},
       **{f'B{m}':(23,19,9,m) for m in ALTERNATIVES},
       'Aordinary':(23,19,11,16),'Bordinary':(23,19,9,11),
       'neither':(23,19,9,16),'both':(23,19,20,21)}
REFERENCES={'A20':'only20','B21':'only21','neither':'neither','both':'both'}


def forced_indices(cell,n,target):
    selected=set(CELLS[cell])
    if len(selected)!=target or any(i<0 or i>=n for i in selected):
        raise ValueError('Forced membership violates the frozen exact-k pool')
    return selected


def solver_frame():
    frame=inspect.currentframe().f_back
    while frame is not None and frame.f_code.co_name!='run_compact_beam_instance':
        frame=frame.f_back
    if frame is None: raise RuntimeError('Native search frame not found')
    return frame


def chain_hits(code,parent,watched):
    seen=set();hits=[]
    while code is not None and code not in seen:
        seen.add(code)
        if code in watched: hits.append(watched[code])
        code=parent.get(code)
    return hits


class SearchTrace:
    """Read-only hooks installed after matching; no extra search or clock queries."""
    def __init__(self,observer,prefix):
        self.observer=observer
        self.watched={bytes.fromhex(c['code']['bytes_hex']):i for i,c in enumerate(prefix['candidates'])}
        self.originals={};self.layers=[];self.expansions=[];self.witnesses=[];self.parent_events=[]
        self.parent_counts=dict(calls=0,reassigned=0,rejected=0);self.final=None

    def install(self):
        replacements={'_record_quality_reservoir_selected_telemetry':self.commit,
            '_assign_parent_link':self.assign_parent,'_completion_witness_grafted_moves':self.graft,
            '_successors_with_n_uncached':self.successors,'CompactBeamResult':self.result}
        for name,replacement in replacements.items():
            self.originals[name]=getattr(beam,name);setattr(beam,name,replacement)

    def restore(self):
        for name,original in self.originals.items(): setattr(beam,name,original)
        self.originals.clear()

    def commit(self,selected_codes,args):
        value=self.originals['_record_quality_reservoir_selected_telemetry'](selected_codes,args)
        loc=solver_frame().f_locals
        candidates=loc['candidates'];parent=loc['parent'];frontier=loc['frontier']
        context_matches=self.observer.context_layer==loc['layer_index']
        core=list(self.observer.layer_context[1]) if context_matches else []
        ranked=self.observer.layer_context[0] if context_matches else []
        ranks={code:rank for rank,(code,data) in enumerate(ranked)}
        selected_set=set(selected_codes);core_set=set(core)
        self.layers.append(dict(layer=loc['layer_index'],candidate_count=len(candidates),
            candidate_fingerprint=fingerprint(candidates),frontier_fingerprint=fingerprint(frontier),
            frontier_count=len(frontier),parent_fingerprint=fingerprint(parent),
            best_depth_fingerprint=fingerprint(loc['best_depth']),core=[c.hex() for c in core],core_observed=context_matches,
            selected=[dict(code=c.hex(),n=candidates[c][0],depth=candidates[c][1],
                parent=candidates[c][2].hex(),qubo=c in loc['selected_by_qubo']) for c in selected_codes],
            watched=[dict(index=i,present=c in candidates,rank=ranks.get(c),core=c in core_set,
                selected=c in selected_set,frontier=any(f[0]==c for f in frontier),
                data=[candidates[c][0],candidates[c][1],candidates[c][2].hex()] if c in candidates else None,
                known_depth=loc['best_depth'].get(c),known_parent=parent[c].hex() if parent.get(c) else None)
                for c,i in self.watched.items()],
            counters={k:loc[k] for k in ('expanded','generated','pruned_bound','pruned_business','dominated','depth_cap')},
            parent_counts=dict(self.parent_counts),
            caches={name:list(function.cache_info()) for name,function in vars(beam).items()
                if callable(function) and hasattr(function,'cache_info')}))
        return value

    def assign_parent(self,parent_chain,child_code,parent_code):
        previous=parent_chain.get(child_code);present=child_code in parent_chain
        result=self.originals['_assign_parent_link'](parent_chain,child_code,parent_code)
        loc=solver_frame().f_locals
        if parent_chain is loc['parent']:
            self.parent_counts['calls']+=1
            self.parent_counts['reassigned']+=bool(result and present and previous!=parent_code)
            self.parent_counts['rejected']+=not result
            if child_code in self.watched:
                self.parent_events.append(dict(layer=loc['layer_index'],index=self.watched[child_code],
                    accepted=result,previous=previous.hex() if previous else None,
                    parent=parent_code.hex() if parent_code else None,previous_present=present))
        return result

    def successors(self,*args,**kwargs):
        caller=inspect.currentframe().f_back.f_code.co_name
        loc=solver_frame().f_locals
        self.expansions.append([loc['layer_index'],caller,args[0].hex(),args[1],loc.get('depth')])
        return self.originals['_successors_with_n_uncached'](*args,**kwargs)

    def graft(self,code,data,parent,instance,rollout,**kwargs):
        result=self.originals['_completion_witness_grafted_moves'](code,data,parent,instance,rollout,**kwargs)
        loc=solver_frame().f_locals
        self.witnesses.append(dict(layer=loc['layer_index'],code=code.hex(),n=data[0],depth=data[1],
            parent=data[2].hex(),accepted=result is not None,moves=len(result) if result is not None else None,
            path_signature=fingerprint([asdict(move) for move in result]) if result is not None else None,
            recorded_rollout_moves=rollout.moves,rollout_status=rollout.status,
            watched_ancestors=chain_hits(data[2],parent,self.watched),
            previous_incumbent_moves=len(loc['rollout_terminal_moves']) if loc['rollout_terminal_moves'] is not None else None))
        return result

    def result(self,*args,**kwargs):
        loc=solver_frame().f_locals
        target=loc['target'];frontier=loc['frontier'];candidates=loc.get('candidates',{})
        clock=self.observer.clock
        reason=('TARGET_FOUND' if target is not None else
                'LOGICAL_DEADLINE' if clock.work_now>=loc['deadline'] else
                'FRONTIER_EMPTY' if not frontier else
                'NO_CANDIDATES' if not candidates else 'OTHER_LOOP_EXIT')
        raw=loc.get('raw_search_codes',[]);search=loc.get('search_path_codes',[])
        self.final=dict(layer=loc['layer_index'],loop_exit_inferred=reason,
            target=target.hex() if target else None,frontier_count=len(frontier),candidate_count=len(candidates),
            depth_cap=loc['depth_cap'],business_cap=loc['cap'],
            raw_search_codes=[c.hex() for c in raw],search_path_codes=[c.hex() for c in search],
            raw_watched_indices=[self.watched[c] for c in raw if c in self.watched],
            parent_counts=dict(self.parent_counts),
            flags={k:loc[k] for k in ('rollout_terminal_source','frg_terminal_witness_used',
                'union_rollout_terminal_graft_used','terminal_bridge_used','frg_seed_used',
                'wang_incumbent_used','compact_lift_repair_inserted')},
            rollout_terminal_moves=len(loc['rollout_terminal_moves']) if loc['rollout_terminal_moves'] is not None else None,
            native_result={k:kwargs.get(k) for k in ('status','visited','open_count','pruned_bound','pruned_business',
                'dominated','depth_reached','frg_terminal_witness_moves','bs_frg_postcompress_moves_saved',
                'qubo_selected_on_search_path','qubo_selected_on_path','qubo_ancestor_on_search_path')})
        return self.originals['CompactBeamResult'](*args,**kwargs)

    def payload(self):
        return dict(schema='native-suffix-provenance/v1',layers=self.layers,expansions=self.expansions,
            witnesses=self.witnesses,parent_events=self.parent_events,final=self.final,
            scope='Native calls only; loop-exit label inferred from final state; no causal cache intervention.')


class ProvenanceObserver(SelectorObserver):
    def __init__(self,clock,expected,cell,mode,trace_enabled):
        super().__init__('sa',capture_limit=0)
        self.clock=clock;self.expected=expected;self.cell=cell;self.mode=mode;self.trace_enabled=trace_enabled
        self.cut=None;self.active=False;self.original_repair=None;self.budget=None;self.target_record=None;self.trace=None
        self.context_layer=None

    def context(self,ranked,core,best_n):
        super().context(ranked,core,best_n)
        self.context_layer=solver_frame().f_locals['layer_index']

    def forced_repair(self,picked,target,linear,features,args,*,stage_callback=None):
        selected=forced_indices(self.cell,len(linear),target)
        if stage_callback is not None:
            stage_callback('forced_membership',selected);stage_callback('swap_scan_skipped',selected)
        return selected

    def sample(self,model,config,usable,target,linear,features,args,feature_sec,build_sec):
        if self.cut is None and len(usable)>target and any(f.is_frg_macro for f in features):
            self.clock.mark_boundary();state,start=boundary_state(args)
            mismatch=[k for k in sorted(set(state)|set(self.expected['state'])) if state.get(k)!=self.expected['state'].get(k)]
            if mismatch: raise RuntimeError('Prefix mismatch: '+json.dumps(mismatch[:30]))
            forced_indices(self.cell,len(usable),target)
            self.cut=dict(case=CASE,call=len(self.records),layer=args._layer_index,n=len(usable),k=target,
                prefix_sec=self.clock.cutoff-start,requested_indices=list(CELLS[self.cell]))
            self.features=features;self.linear=linear
            self.clock.continue_live()
            if self.mode=='work': self.budget=WorkBudget(self.clock,args);self.budget.activate()
            if self.trace_enabled:
                self.trace=SearchTrace(self,self.expected);self.trace.install()
            self.original_repair=beam._repair_union_diversity_selection
            beam._repair_union_diversity_selection=self.forced_repair;self.active=True
        return super().sample(model,config,usable,target,linear,features,args,feature_sec,build_sec)

    def finish(self,ordered,selector_sec):
        super().finish(ordered,selector_sec)
        if self.active:
            if set(ordered)!=set(CELLS[self.cell]): raise RuntimeError('Native ordering changed membership')
            args=inspect.currentframe().f_back.f_locals['args']
            self.target_record=dict(self.current)
            self.target_record.update(forced_membership=True,original_swap_scan_completed=False,
                macro_selected=sum(self.features[i].is_frg_macro for i in ordered),
                original_proxy_objective=beam._union_diversity_objective(set(ordered),self.linear,self.features,args))
            beam._repair_union_diversity_selection=self.original_repair;self.original_repair=None;self.active=False

    def restore(self):
        if self.original_repair is not None:
            beam._repair_union_diversity_selection=self.original_repair;self.original_repair=None
        if self.trace is not None: self.trace.restore()
        if self.budget is not None: self.budget.restore()


def freeze():
    if (OUT/'manifest.json').exists(): raise FileExistsError('Protocol already frozen')
    parent=round7_manifest()
    prefix=ROOT/parent['prefix']['path'];clock=ROOT/parent['clock']['path']
    data=json.loads(prefix.read_text(encoding='utf-8'))
    macros=[i for i,c in enumerate(data['candidates']) if c['feature']['is_frg_macro']]
    assert tuple(i for i in macros if i not in (23,19))==ALTERNATIVES
    ordinary=[i for i,c in enumerate(data['candidates']) if i not in (9,16) and not c['feature']['is_frg_macro']
        and c['feature']['n_value']==14 and c['feature']['depth']==14]
    assert min(ordinary,key=lambda i:(data['linear'][i],i))==11
    sources=['scripts/production_route_provenance.py','scripts/production_route_membership.py',
        'scripts/production_selector_factorial.py','scripts/production_selector_replay.py',
        'scripts/benchmark_compact_beam_qubo_n0.py','scripts/beam_selector_research.py']
    write_json(OUT/'manifest.json',dict(date='2026-10-02',case=CASE,cells=CELLS,macro_alternatives=ALTERNATIVES,
        ordinary_placebo=11,ordinary_selection='Minimum frozen unary ordinary excluding 9/16 at same N=14, depth=14',
        prefix=parent['prefix'],clock=parent['clock'],source_sha256={p:digest(ROOT/p) for p in sources},
        work_settings=WORK_SETTINGS,successor_limit=SUCCESSOR_LIMIT,work_repeats=2,wall_repeats=3,
        tracing='Work r0 only, after full matching; work r1 and all wall runs uninstrumented',
        reference_cells=REFERENCES,watchdog_sec=120,
        repeat_gate='Exact path/work/telemetry/selection signature equality; trace neutrality and old memberships',
        scope='One frozen C2 boundary; exhaustive remaining macro identities and two predefined ordinary placebos; NO_PROMOTION'))


def manifest():
    data=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'));round7_manifest()
    for p,sha in data['source_sha256'].items():
        if digest(ROOT/p)!=sha: raise ValueError('Frozen source changed: '+p)
    for key in ('prefix','clock'):
        item=data[key]
        if digest(ROOT/item['path'])!=item['sha256']: raise ValueError('Frozen prefix changed')
    return data


def worker(cell,mode,repeat):
    m=manifest();instance=load_instance(next(c for c in business_manifest()['cases'] if c['case']==CASE)['instance'])
    args=business_args();expected=json.loads((ROOT/m['prefix']['path']).read_text(encoding='utf-8'))
    with gzip.open(ROOT/m['clock']['path'],'rb') as stream:
        values=array('d');values.frombytes(stream.read())
    if expected['byteorder']!=sys.byteorder: values.byteswap()
    random.seed(37);np.random.seed(37)
    clock=SuffixClock(values,expected['cutoff'],mode)
    trace_enabled=mode=='work' and repeat==0
    observer=ProvenanceObserver(clock,expected,cell,mode,trace_enabled);args._union_selector_observer=observer
    start=REAL_TIME();time.perf_counter=clock
    try: result=beam.run_compact_beam_instance(CASE,instance,args)
    finally:
        time.perf_counter=REAL_TIME;observer.restore()
    elapsed=REAL_TIME()-start
    if observer.cut is None: raise RuntimeError('Matched boundary not reached')
    business_ok,goal_ok,message=beam._validate_instance_result(result,instance,business_alpha=1.3,business_cap_enabled=True)
    path=[asdict(move) for move in result.moves]
    work=dict(observer.budget.stats) if observer.budget is not None else {}
    telemetry={k:getattr(result,k) for k in ('expanded','generated','qubo_calls','post_polish_attempts',
        'post_polish_improvements','post_polish_moves_saved','post_polish_suffix_expanded',
        'post_polish_window_attempts','post_polish_bridge_expanded') if hasattr(result,k)}
    trace=[dict(call=r['call'],n=r['n'],k=r['k'],selected=r['selected']) for r in observer.records]
    signature=fingerprint(dict(path=path,work=work,telemetry=telemetry,selection_trace=trace))
    reference_match=None
    if mode=='work' and cell in REFERENCES:
        ref=json.loads((ROUND7/'runs'/f'work__{REFERENCES[cell]}__r0.json').read_text(encoding='utf-8'))
        reference_match=signature==ref['row']['signature']
        if not reference_match: raise RuntimeError('Old membership control failed')
    row=dict(case=CASE,cell=cell,mode=mode,repeat=repeat,traced=trace_enabled,state_matched=True,
        valid=business_ok and goal_ok,business_ok=business_ok,goal_ok=goal_ok,moves=len(path) if goal_ok else None,
        selected=observer.target_record['selected'],macro_selected=observer.target_record['macro_selected'],
        original_proxy_objective=observer.target_record['original_proxy_objective'],physical_solver_sec=elapsed,
        charged_time_sec=result.runtime_sec if mode=='wall' else None,
        logical_within_budget=result.runtime_sec<=20 if mode=='wall' else None,
        reference_signature_matched=reference_match,signature=signature,validation_message=message,
        **telemetry,**work)
    extras={k:getattr(result,k) for k in ('status','pruned_bound','pruned_business','dominated','depth_reached',
        'visited','open_count','frg_terminal_witness_used','frg_terminal_witness_moves','frg_seed_used',
        'wang_incumbent_used','union_rollout_terminal_graft_used','bs_frg_postcompress_moves_saved',
        'qubo_selected_on_search_path','qubo_selected_on_path','qubo_ancestor_on_search_path')}
    stem=f'{mode}__{cell}__r{repeat}'
    write_json(OUT/'runs'/f'{stem}.json',dict(row=row,path=path,cut=observer.cut,target=observer.target_record,
        selection_trace=trace,result_metadata=extras))
    if observer.trace is not None:
        trace_path=OUT/'traces'/f'{stem}.json.gz';trace_path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(trace_path,'wt',encoding='utf-8') as stream:
            json.dump(observer.trace.payload(),stream,ensure_ascii=False,separators=(',',':'))
    if mode=='work' and repeat==1:
        first=json.loads((OUT/'runs'/f'work__{cell}__r0.json').read_text(encoding='utf-8'))
        if signature!=first['row']['signature']: raise RuntimeError('Tracing neutrality/repetition gate failed: '+cell)
    print(json.dumps(row,ensure_ascii=False),flush=True)


def audit_work():
    audit=[]
    for cell in CELLS:
        rows=[json.loads((OUT/'runs'/f'work__{cell}__r{i}.json').read_text(encoding='utf-8'))['row'] for i in range(2)]
        if not all(r['state_matched'] for r in rows) or rows[0]['signature']!=rows[1]['signature']:
            raise RuntimeError('Work-state or trace-neutrality gate failed: '+cell)
        if cell in REFERENCES and not all(r['reference_signature_matched'] for r in rows):
            raise RuntimeError('Old control failed: '+cell)
        if not (OUT/'traces'/f'work__{cell}__r0.json.gz').exists(): raise RuntimeError('Trace missing: '+cell)
        audit.append(dict(cell=cell,trace_neutral=True,signatures_equal=True,
            reference_matched=True if cell in REFERENCES else None,signature=rows[0]['signature']))
    write_json(OUT/'work_repeat_audit.json',audit)
    return audit


def panel(mode):
    m=manifest();cells=list(REFERENCES)+[cell for cell in CELLS if cell not in REFERENCES]
    if mode=='wall': audit_work()
    for repeat in range(m[mode+'_repeats']):
        offset=repeat%len(cells)
        for cell in cells[offset:]+cells[:offset]:
            stem=f'{mode}__{cell}__r{repeat}'
            if (OUT/'runs'/f'{stem}.json').exists(): continue
            cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--mode',mode,'--cell',cell,'--repeat',str(repeat)]
            try:
                proc=subprocess.run(cmd,cwd=ROOT,env={**os.environ,**ENVIRONMENT},capture_output=True,
                    text=True,encoding='utf-8',errors='replace',timeout=m['watchdog_sec'])
            except subprocess.TimeoutExpired:
                write_json(OUT/'runs'/f'{stem}.json',dict(row=dict(case=CASE,cell=cell,mode=mode,repeat=repeat,
                    valid=False,moves=None,state_matched=False,status='EXTERNAL_WATCHDOG'),path=[]))
                print(stem+' EXTERNAL_WATCHDOG',flush=True);continue
            if proc.returncode:
                write_json(OUT/'errors'/f'{stem}.json',dict(stdout=proc.stdout,stderr=proc.stderr,returncode=proc.returncode))
                raise RuntimeError(proc.stderr)
            print(proc.stdout.strip(),flush=True)
    rows=[json.loads(p.read_text(encoding='utf-8'))['row'] for p in sorted((OUT/'runs').glob('*.json'))]
    write_csv(OUT/'results.csv',rows)
    if mode=='work': audit_work()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run','worker'])
    parser.add_argument('--mode',choices=['work','wall'],default='work');parser.add_argument('--cell',choices=CELLS,default='A20')
    parser.add_argument('--repeat',type=int,default=0);args=parser.parse_args()
    if args.command=='freeze': freeze()
    elif args.command=='run': panel(args.mode)
    else: worker(args.cell,args.mode,args.repeat)

if __name__=='__main__': main()
