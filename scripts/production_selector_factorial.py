"""Single-decision pair x quota factorial, native-time and fixed-work suffixes."""
from __future__ import annotations
import argparse
from array import array
import copy
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
from beam_selector_research import PAIR_KEYS,SelectorObserver,decode_feature,initial_set
from probe_business_macro_windows import business_args,load_instance,manifest as business_manifest
from production_selector_replay import PrefixClock,REAL_TIME,boundary_state,fingerprint,REPLAY,manifest as replay_manifest
from run_beam_selector_research import ENVIRONMENT,digest,write_csv,write_json

OUT=ROOT/'docs/experiments/beam_selector_round6_20260924'
FACTORS={'none':(False,False),'pair_only':(True,False),'quota_only':(False,True),'full':(True,True)}
WORK_SETTINGS={'frg_rollout_max_generated':512,'post_polish_max_attempts':2048,
               'inner_jobs':1,'post_polish_inner_jobs':1}
SUCCESSOR_LIMIT=5_000_000


def set_factors(args,factor):
    pair,quota=FACTORS[factor]
    keys=PAIR_KEYS+['union_diversity_qubo_adaptive_macro_quota_enabled']
    saved={key:(hasattr(args,key),getattr(args,key,None)) for key in keys}
    if not pair:
        for key in PAIR_KEYS: setattr(args,key,0.0)
    if not quota: args.union_diversity_qubo_adaptive_macro_quota_enabled=False
    return saved


def restore_factors(args,saved):
    for key,(existed,value) in saved.items():
        if existed: setattr(args,key,value)
        elif hasattr(args,key): delattr(args,key)


class SuffixClock(PrefixClock):
    def __init__(self,values,cutoff,mode):
        super().__init__(values,cutoff)
        self.mode=mode;self.work_now=cutoff

    def __call__(self):
        if self.live and self.mode=='work': return self.work_now
        return super().__call__()


class WorkBudget:
    """Count real generated successors; a reached ceiling expires search cooperatively."""
    def __init__(self,clock,args,limit=SUCCESSOR_LIMIT):
        self.clock=clock;self.args=args;self.limit=limit
        self.original_successors=None;self.original_frg=None
        self.stats=dict(uncached_successors=0,successor_calls=0,work_guard_reached=False,
            frg_calls=0,frg_generated=0,frg_max_batch_total=0,frg_over_threshold_calls=0)

    def expire(self):
        self.stats['work_guard_reached']=True
        self.args.strict_search_deadline=True
        self.clock.work_now=max(self.clock.work_now,float(self.args._overall_search_deadline)+1.0)

    def successors(self,*args,**kwargs):
        self.stats['successor_calls']+=1
        if self.stats['work_guard_reached']: return
        iterator=iter(self.original_successors(*args,**kwargs))
        while True:
            if self.stats['uncached_successors']>=self.limit:
                self.expire();return
            try: candidate=next(iterator)
            except StopIteration: return
            self.stats['uncached_successors']+=1
            yield candidate

    def frg(self,*args,**kwargs):
        result=self.original_frg(*args,**kwargs)
        self.stats['frg_calls']+=1
        generated=int(result.generated)
        self.stats['frg_generated']+=generated
        self.stats['frg_max_batch_total']=max(self.stats['frg_max_batch_total'],generated)
        self.stats['frg_over_threshold_calls']+=generated>512
        return result

    def activate(self):
        for key,value in WORK_SETTINGS.items(): setattr(self.args,key,value)
        self.original_successors=beam._successors_with_n_uncached
        self.original_frg=beam._bs_frg_completion_cached
        beam._successors_with_n_uncached=self.successors
        beam._bs_frg_completion_cached=self.frg

    def restore(self):
        if self.original_successors is not None:
            beam._successors_with_n_uncached=self.original_successors
            beam._bs_frg_completion_cached=self.original_frg


class FactorialObserver(SelectorObserver):
    def __init__(self,clock,expected,factor,mode):
        super().__init__('sa',capture_limit=0)
        self.clock=clock;self.expected=expected;self.factor=factor;self.mode=mode
        self.cut=None;self.cut_active=False;self.saved=None;self.target_record=None;self.budget=None

    def sample(self,model,config,usable,target,linear,features,args,feature_sec,build_sec):
        if self.cut is None and len(usable)>target and any(f.is_frg_macro for f in features):
            self.clock.mark_boundary()
            state,start=boundary_state(args)
            mismatch=[k for k in sorted(set(state)|set(self.expected['state'])) if state.get(k)!=self.expected['state'].get(k)]
            if mismatch: raise RuntimeError('Prefix mismatch: '+json.dumps(mismatch[:30]))
            self.cut=dict(layer=args._layer_index,call=len(self.records),n=len(usable),k=target,
                prefix_sec=self.clock.cutoff-start,
                quota_target_original=beam._union_diversity_adaptive_macro_quota_target(features,target,args))
            self.features=features;self.linear=linear
            self.saved=set_factors(args,self.factor)
            # Top-k is exactly the marginal-greedy specialization of the linear cell.
            self.arm='linear' if self.factor=='none' else 'marginal'
            self.cut_active=True
            self.clock.continue_live()
            if self.mode=='work':
                self.budget=WorkBudget(self.clock,args)
                self.budget.activate()
        return super().sample(model,config,usable,target,linear,features,args,feature_sec,build_sec)

    def finish(self,ordered,selector_sec):
        super().finish(ordered,selector_sec)
        if self.cut_active:
            args=inspect.currentframe().f_back.f_locals['args']
            selected=set(ordered)
            self.target_record=dict(self.current)
            self.target_record['macro_selected']=sum(self.features[i].is_frg_macro for i in selected)
            self.target_record['linear_sum']=sum(self.linear[i] for i in selected)
            self.target_record['quota_value']=beam._union_diversity_adaptive_macro_quota_penalty(selected,self.features,args)
            self.target_record['objective']=beam._union_diversity_objective(selected,self.linear,self.features,args)
            self.target_record['generic_pair_value']=self.target_record['objective']-self.target_record['linear_sum']-self.target_record['quota_value']
            restore_factors(args,self.saved)
            self.saved=None;self.arm='sa';self.cut_active=False


def freeze():
    if (OUT/'manifest.json').exists(): raise FileExistsError('Factorial already frozen')
    previous=replay_manifest()
    sources=['scripts/production_selector_factorial.py','scripts/production_selector_replay.py',
        'scripts/benchmark_compact_beam_qubo_n0.py','scripts/beam_selector_research.py']
    inputs=[]
    for case in previous['cases']:
        for suffix in ('.json','.clock.gz'):
            p=REPLAY/'prefixes'/f'{case}{suffix}'
            inputs.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p)))
    write_json(OUT/'manifest.json',dict(cases=previous['cases'],factors=FACTORS,
        source_sha256={p:digest(ROOT/p) for p in sources},prefixes=inputs,
        work_settings=WORK_SETTINGS,uncached_successor_limit=SUCCESSOR_LIMIT,
        work_repeats=2,wall_repeats=3,watchdog_sec=120,
        replacement='Only one matched selector; later factors restored. All cells use objective-consistent greedy/repair.',
        work_contract='Prefix clock replay then frozen logical timer, explicit probe/polish/uncached-successor limits. Physical watchdog is a failed run, not a solved result.',
        time_contract='Original real-clock suffix, 20-second nominal request; physical and charged time separate.',
        repeat_gate='Exact semantic signature equality across work repetitions, including final moves and work counts.',
        selection_of_cases='All four round-5 matched production prefixes; no quality-based subset or weight tuning.'))


def manifest():
    m=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    replay_manifest()
    correction_path=OUT/'source_correction.json'
    corrections=json.loads(correction_path.read_text(encoding='utf-8')) if correction_path.exists() else {}
    for path,sha in m['source_sha256'].items():
        actual=digest(ROOT/path);correction=corrections.get(path,{})
        allowed=correction.get('original_sha256')==sha and correction.get('corrected_sha256')==actual
        if actual!=sha and not allowed: raise ValueError(f'Frozen source changed: {path}')
    for item in m['prefixes']:
        if digest(ROOT/item['path'])!=item['sha256']: raise ValueError('Frozen prefix changed')
    return m


def static_decisions():
    m=manifest();rows=[]
    for case in m['cases']:
        prefix=json.loads((REPLAY/'prefixes'/f'{case}.json').read_text(encoding='utf-8'))
        features=[decode_feature(c['feature']) for c in prefix['candidates']]
        for factor in FACTORS:
            args=business_args();args._quality_reservoir_objective_telemetry_enabled=False
            set_factors(args,factor)
            picked=initial_set('linear' if factor=='none' else 'marginal',prefix['linear'],features,args,prefix['k'],37)
            final=beam._repair_union_diversity_selection(picked,prefix['k'],prefix['linear'],features,args)
            ordered=sorted(final,key=lambda i:(beam._union_diversity_objective({i},prefix['linear'],features,args),
                features[i].n_value,features[i].depth,bytes.fromhex(prefix['candidates'][i]['code']['bytes_hex'])))
            rows.append(dict(case=case,factor=factor,selected=ordered,
                macro_selected=sum(features[i].is_frg_macro for i in final),
                linear_sum=sum(prefix['linear'][i] for i in final),
                quota_value=beam._union_diversity_adaptive_macro_quota_penalty(final,features,args),
                objective=beam._union_diversity_objective(final,prefix['linear'],features,args)))
    write_csv(OUT/'static_decisions.csv',rows)
    print(json.dumps(rows,ensure_ascii=False,indent=2))


def worker(case,factor,mode,repeat):
    manifest();bm=business_manifest();data=next(c for c in bm['cases'] if c['case']==case)
    instance=load_instance(data['instance']);args=business_args()
    expected=json.loads((REPLAY/'prefixes'/f'{case}.json').read_text(encoding='utf-8'))
    with gzip.open(REPLAY/'prefixes'/f'{case}.clock.gz','rb') as stream:
        values=array('d');values.frombytes(stream.read())
    if expected['byteorder']!=sys.byteorder: values.byteswap()
    random.seed(37);np.random.seed(37)
    clock=SuffixClock(values,expected['cutoff'],mode)
    observer=FactorialObserver(clock,expected,factor,mode)
    args._union_selector_observer=observer
    started=REAL_TIME();time.perf_counter=clock
    try:
        result=beam.run_compact_beam_instance(case,instance,args)
    finally:
        time.perf_counter=REAL_TIME
        if observer.budget is not None: observer.budget.restore()
    physical=REAL_TIME()-started
    if observer.cut is None: raise RuntimeError('Matched boundary was not reached')
    business_ok,goal_ok,message=beam._validate_instance_result(result,instance,business_alpha=1.3,business_cap_enabled=True)
    path=[asdict(move) for move in result.moves]
    work=dict(observer.budget.stats) if observer.budget is not None else {}
    telemetry={k:getattr(result,k) for k in ('expanded','generated','qubo_calls','post_polish_attempts',
        'post_polish_improvements','post_polish_moves_saved','post_polish_suffix_expanded',
        'post_polish_window_attempts','post_polish_bridge_expanded') if hasattr(result,k)}
    selection_trace=[dict(call=r['call'],n=r['n'],k=r['k'],selected=r['selected']) for r in observer.records]
    signature=fingerprint(dict(path=path,work=work,telemetry=telemetry,selection_trace=selection_trace))
    row=dict(case=case,factor=factor,mode=mode,repeat=repeat,state_matched=True,
        valid=business_ok and goal_ok,business_ok=business_ok,goal_ok=goal_ok,moves=len(path) if goal_ok else None,
        physical_solver_sec=physical,charged_time_sec=result.runtime_sec if mode=='wall' else None,
        logical_within_budget=result.runtime_sec<=20 if mode=='wall' else None,
        boundary_selected=observer.target_record['selected'],macro_selected=observer.target_record['macro_selected'],
        boundary_objective=observer.target_record['objective'],signature=signature,validation_message=message,
        **telemetry,**work)
    write_json(OUT/'runs'/f'{case}__{mode}__{factor}__r{repeat}.json',
        dict(row=row,path=path,boundary=observer.cut,target=observer.target_record,selection_trace=selection_trace))
    print(json.dumps(row,ensure_ascii=False),flush=True)


def panel(mode,case_filter=None):
    m=manifest();cases=[c for c in m['cases'] if not case_filter or c==case_filter]
    repeats=m[mode+'_repeats'];factors=list(FACTORS)
    for ci,case in enumerate(cases):
        for repeat in range(repeats):
            offset=(ci+repeat)%len(factors)
            for factor in factors[offset:]+factors[:offset]:
                stem=f'{case}__{mode}__{factor}__r{repeat}'
                if (OUT/'runs'/f'{stem}.json').exists(): continue
                cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--mode',mode,'--case',case,'--factor',factor,'--repeat',str(repeat)]
                try:
                    proc=subprocess.run(cmd,cwd=ROOT,env={**os.environ,**ENVIRONMENT},capture_output=True,
                        text=True,encoding='utf-8',errors='replace',timeout=m['watchdog_sec'])
                except subprocess.TimeoutExpired:
                    write_json(OUT/'runs'/f'{stem}.json',dict(row=dict(case=case,mode=mode,factor=factor,repeat=repeat,
                        valid=False,moves=None,state_matched=False,status='EXTERNAL_WATCHDOG'),path=[]))
                    print(stem+' EXTERNAL_WATCHDOG',flush=True);continue
                if proc.returncode:
                    write_json(OUT/'errors'/f'{stem}.json',dict(stdout=proc.stdout,stderr=proc.stderr,returncode=proc.returncode))
                    raise RuntimeError(proc.stderr)
                print(proc.stdout.strip(),flush=True)
    rows=[json.loads(p.read_text(encoding='utf-8'))['row'] for p in sorted((OUT/'runs').glob('*.json'))]
    write_csv(OUT/'results.csv',rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','static','run','worker'])
    p.add_argument('--mode',choices=['work','wall'],default='work');p.add_argument('--case')
    p.add_argument('--factor',choices=FACTORS,default='full');p.add_argument('--repeat',type=int,default=0)
    ns=p.parse_args()
    if ns.command=='freeze': freeze()
    elif ns.command=='static': static_decisions()
    elif ns.command=='run': panel(ns.mode,ns.case)
    else: worker(ns.case,ns.factor,ns.mode,ns.repeat)

if __name__=='__main__': main()
