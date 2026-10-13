"""Replay a production prefix with recorded clocks, then replace one selector.

No continuation is accepted unless all recorded prefix clock calls are consumed
and the live solver/caller state fingerprints match the captured boundary.
"""
from __future__ import annotations
import argparse
from array import array
import gzip
import hashlib
import inspect
import json
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

REAL_TIME=time.perf_counter
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'scripts',ROOT/'src'):
    sys.path.insert(0,str(p))
import numpy as np
import benchmark_compact_beam_qubo_n0 as beam
from beam_selector_research import PAIR_KEYS,SelectorObserver,encode
from probe_business_macro_windows import business_args,load_instance,manifest as business_manifest
from probe_production_macro_windows import OUT
from run_beam_selector_research import ENVIRONMENT,digest,write_json,write_csv

REPLAY=OUT/'same_prefix'
STRATEGIES=('native','linear_model','greedy_full')


class PrefixClock:
    def __init__(self,values=None,cutoff=None,real_clock=REAL_TIME):
        self.replay=values is not None
        self.values=array('d',values or [])
        self.position=0
        self.cutoff=cutoff
        self.real_clock=real_clock
        self.live=False
        self.anchor=0.0

    def __call__(self):
        if self.live:
            return self.cutoff+self.real_clock()-self.anchor
        if self.replay:
            if self.position>=len(self.values):
                raise RuntimeError('Prefix requested more clock calls than captured')
            value=self.values[self.position];self.position+=1
            return value
        value=self.real_clock()
        self.values.append(value)
        return value

    def mark_boundary(self):
        if self.replay:
            if self.position!=len(self.values):
                raise RuntimeError('Prefix reached selector before consuming its clock trace')
        else:
            self.cutoff=self.real_clock()

    def continue_live(self):
        self.anchor=self.real_clock()
        self.live=True


def fingerprint(value):
    return hashlib.sha256(pickle.dumps(value,protocol=5)).hexdigest()


def boundary_state(args):
    """Fingerprint data in every active search frame and all mutable args state."""
    result={}
    frame=inspect.currentframe().f_back
    while frame is not None and frame.f_code.co_name!='_select_union_diversity_qubo_supplement':
        frame=frame.f_back
    index=0;found=False;solver_start=None
    while frame is not None:
        name=frame.f_code.co_name
        for key,value in frame.f_locals.items():
            if key in ('args','observer') or callable(value):
                continue
            result[f'frame{index}:{name}:{key}']=fingerprint(value)
        if name=='run_compact_beam_instance':
            found=True;solver_start=frame.f_locals['start'];break
        frame=frame.f_back;index+=1
    if not found:
        raise RuntimeError('Production search frame not found')
    for key,value in vars(args).items():
        if key=='_union_selector_observer': continue
        if callable(value):
            raise TypeError(f'Unexpected callable argument state: {key}')
        result['args:'+key]=fingerprint(value)
    for key,value in vars(beam).items():
        if callable(value) and hasattr(value,'cache_info'):
            result['cache:'+key]=fingerprint(tuple(value.cache_info()))
    result['rng:python']=fingerprint(random.getstate())
    result['rng:numpy']=fingerprint(np.random.get_state())
    return result,solver_start


class ReplayObserver(SelectorObserver):
    def __init__(self,clock,strategy,expected=None):
        super().__init__('sa',capture_limit=0)
        self.clock=clock;self.strategy=strategy;self.expected=expected
        self.cut=None;self.saved_weights=None;self.target_record=None
        self.cut_active=False

    def sample(self,model,config,usable,target,linear,features,args,feature_sec,build_sec):
        macro_count=sum(bool(f.is_frg_macro) for f in features)
        if self.cut is None and len(usable)>target and macro_count>0:
            self.clock.mark_boundary()
            started=REAL_TIME()
            state,solver_start=boundary_state(args)
            if self.expected is not None:
                all_keys=set(state)|set(self.expected['state'])
                mismatch=[k for k in sorted(all_keys) if state.get(k)!=self.expected['state'].get(k)]
                if mismatch:
                    raise RuntimeError('Prefix state mismatch: '+json.dumps(mismatch[:30]))
            self.cut=dict(call=len(self.records),layer=int(getattr(args,'_layer_index',-1)),n=len(usable),k=target,
                macro_count=macro_count,clock_calls=len(self.clock.values),cutoff=self.clock.cutoff,
                solver_start=solver_start,prefix_sec=self.clock.cutoff-solver_start,state=state,
                instrumentation_sec=REAL_TIME()-started,
                candidates=encode([dict(code=c,data=data,feature=f) for (c,data),f in zip(usable,features)]),
                linear=linear,model=model.to_dict())
            if self.strategy=='linear_model':
                self.saved_weights={k:(hasattr(args,k),getattr(args,k,None)) for k in PAIR_KEYS+['union_diversity_qubo_adaptive_macro_quota_enabled']}
                for key in PAIR_KEYS: setattr(args,key,0.0)
                args.union_diversity_qubo_adaptive_macro_quota_enabled=False
                self.arm='linear'
            elif self.strategy=='greedy_full':
                self.arm='marginal'
            self.cut_active=True
            self.clock.continue_live()
        return super().sample(model,config,usable,target,linear,features,args,feature_sec,build_sec)

    def finish(self,ordered,selector_sec):
        super().finish(ordered,selector_sec)
        if self.cut_active:
            self.target_record=dict(self.current)
            # Restore only after native caller repair, telemetry and output sorting.
            frame=inspect.currentframe().f_back
            args=frame.f_locals['args']
            if self.saved_weights is not None:
                for key,(existed,value) in self.saved_weights.items():
                    if existed: setattr(args,key,value)
                    else: delattr(args,key)
            self.saved_weights=None;self.arm='sa';self.cut_active=False


def freeze():
    path=REPLAY/'manifest.json'
    if path.exists(): raise FileExistsError('Replay contract already frozen')
    m=business_manifest()
    eligible=[]
    for c in m['cases']:
        row=json.loads((OUT/'business_runs'/f"{c['case']}.json").read_text(encoding='utf-8'))['row']
        if row['eligible_calls']>0: eligible.append(c['case'])
    sources=['scripts/production_selector_replay.py','scripts/probe_business_macro_windows.py',
        'scripts/benchmark_compact_beam_qubo_n0.py','scripts/beam_selector_research.py']
    write_json(path,dict(cases=eligible,strategies=list(STRATEGIES),cap_sec=20,seed=37,
        repeats=3,
        source_sha256={p:digest(ROOT/p) for p in sources},
        boundary='First naturally nontrivial union window with macro features in each fresh recorded prefix',
        state_check='All active solver-frame data, mutable args/caches, pure-cache counts and Python/numpy RNG; exact prefix clock consumption',
        replacement='One selector only; all later selectors and generators are native',
        timing='Prefix logical clock is replayed; after cut real continuation time advances it. Report physical solve time separately. No speed claim.',
        gate='Stop a replay on state/clock mismatch. Do not treat independently reconstructed unequal states as continuations.',
        source_environment=ENVIRONMENT))


def manifest():
    m=json.loads((REPLAY/'manifest.json').read_text(encoding='utf-8'))
    business_manifest()
    for p,sha in m['source_sha256'].items():
        if digest(ROOT/p)!=sha: raise ValueError(f'Source changed: {p}')
    return m


def worker(case_id,strategy,capture=False,repeat=0):
    manifest();bm=business_manifest()
    c=next(c for c in bm['cases'] if c['case']==case_id)
    instance=load_instance(c['instance']);args=business_args()
    random.seed(37);np.random.seed(37)
    expected=None
    if capture:
        clock=PrefixClock()
    else:
        expected=json.loads((REPLAY/'prefixes'/f'{case_id}.json').read_text(encoding='utf-8'))
        with gzip.open(REPLAY/'prefixes'/f'{case_id}.clock.gz','rb') as stream:
            values=array('d');values.frombytes(stream.read())
        if expected['byteorder']!=sys.byteorder: values.byteswap()
        clock=PrefixClock(values,expected['cutoff'])
    observer=ReplayObserver(clock,strategy,expected)
    args._union_selector_observer=observer
    started=REAL_TIME()
    time.perf_counter=clock
    try:
        result=beam.run_compact_beam_instance(case_id,instance,args)
    finally:
        time.perf_counter=REAL_TIME
    physical_wall=REAL_TIME()-started
    business_ok,goal_ok,msg=beam._validate_instance_result(result,instance,business_alpha=1.3,business_cap_enabled=True)
    if observer.cut is None:
        write_json(REPLAY/'errors'/f'{case_id}__{strategy}.json',dict(error='NO_ELIGIBLE_BOUNDARY',physical_wall_sec=physical_wall))
        raise RuntimeError('No eligible macro boundary in captured trajectory')
    if capture:
        prefix=dict(observer.cut,byteorder=sys.byteorder)
        write_json(REPLAY/'prefixes'/f'{case_id}.json',prefix)
        with gzip.open(REPLAY/'prefixes'/f'{case_id}.clock.gz','wb') as stream: stream.write(clock.values.tobytes())
    if not capture and strategy=='native':
        baseline=json.loads((REPLAY/'results'/f'{case_id}__capture.json').read_text(encoding='utf-8'))
        if observer.target_record['selected']!=baseline['row']['selected']:
            raise RuntimeError('Native selector did not reproduce its captured choice')
    row=dict(case=case_id,strategy=strategy,capture=capture,repeat=repeat,state_matched=not capture,
        valid=business_ok and goal_ok,business_ok=business_ok,goal_ok=goal_ok,moves=len(result.moves) if goal_ok else None,
        physical_solver_sec=physical_wall,charged_prefix_plus_continuation_sec=result.runtime_sec,
        within_logical_budget=result.runtime_sec<=20,prefix_sec=observer.cut['prefix_sec'],
        boundary_call=observer.cut['call'],boundary_layer=observer.cut['layer'],
        n=observer.cut['n'],k=observer.cut['k'],macro_count=observer.cut['macro_count'],
        selected=observer.target_record['selected'],sampler_raw_count_error=observer.target_record['raw_count_error'],
        validation_message=msg)
    from dataclasses import asdict
    suffix='capture' if capture else f'{strategy}__r{repeat}'
    write_json(REPLAY/'results'/f'{case_id}__{suffix}.json',dict(row=row,path=[asdict(move) for move in result.moves],target_record=observer.target_record))
    print(json.dumps(row,ensure_ascii=False),flush=True)


def run_panel():
    m=manifest()
    for ci,case in enumerate(m['cases']):
        schedule=[('capture',0)]
        for repeat in range(m['repeats']):
            offset=(ci+repeat)%len(STRATEGIES)
            schedule.extend((strategy,repeat) for strategy in STRATEGIES[offset:]+STRATEGIES[:offset])
        for strategy,repeat in schedule:
            suffix='capture' if strategy=='capture' else f'{strategy}__r{repeat}'
            if (REPLAY/'results'/f'{case}__{suffix}.json').exists(): continue
            command=[sys.executable,str(Path(__file__).resolve()),'worker','--case',case,'--strategy','native' if strategy=='capture' else strategy]
            if strategy=='capture': command.append('--capture')
            command.extend(['--repeat',str(repeat)])
            proc=subprocess.run(command,cwd=ROOT,env={**os.environ,**ENVIRONMENT},capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
            if proc.returncode:
                write_json(REPLAY/'errors'/f'{case}__{suffix}.json',dict(returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
                raise RuntimeError(proc.stderr)
            print(proc.stdout.strip(),flush=True)
    rows=[json.loads(p.read_text(encoding='utf-8'))['row'] for p in sorted((REPLAY/'results').glob('*.json'))]
    write_csv(REPLAY/'results.csv',rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run','worker'])
    p.add_argument('--case');p.add_argument('--strategy',choices=STRATEGIES,default='native');p.add_argument('--capture',action='store_true')
    p.add_argument('--repeat',type=int,default=0)
    ns=p.parse_args()
    if ns.command=='freeze': freeze()
    elif ns.command=='run': run_panel()
    else: worker(ns.case,ns.strategy,ns.capture,ns.repeat)

if __name__=='__main__': main()
