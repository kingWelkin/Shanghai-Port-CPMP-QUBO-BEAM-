"""Force one frozen route-membership choice, then resume the native production policy."""
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
from production_selector_factorial import SuffixClock,WorkBudget,WORK_SETTINGS,SUCCESSOR_LIMIT,manifest as factorial_manifest
from run_beam_selector_research import ENVIRONMENT,digest,write_csv,write_json

OUT=ROOT/'docs/experiments/beam_selector_round7_20261002'
ROUND6=ROOT/'docs/experiments/beam_selector_round6_20260924'
CASE='C2_26_single'
CELLS={'neither':(23,19,9,16),'only20':(23,19,20,16),
       'only21':(23,19,9,21),'both':(23,19,20,21)}


def forced_indices(cell,n,target):
    indices=set(CELLS[cell])
    if len(indices)!=target or any(i<0 or i>=n for i in indices):
        raise ValueError('Forced membership violates the frozen exact-k pool')
    return indices


class MembershipObserver(SelectorObserver):
    def __init__(self,clock,expected,cell,mode):
        super().__init__('sa',capture_limit=0)
        self.clock=clock;self.expected=expected;self.cell=cell;self.mode=mode
        self.cut=None;self.active=False;self.original_repair=None;self.budget=None;self.target_record=None

    def forced_repair(self,picked,target,linear,features,args,*,stage_callback=None):
        selected=forced_indices(self.cell,len(linear),target)
        if stage_callback is not None:
            stage_callback('forced_membership',selected)
            stage_callback('swap_scan_skipped',selected)
        return selected

    def restore_repair(self):
        if self.original_repair is not None:
            beam._repair_union_diversity_selection=self.original_repair
            self.original_repair=None

    def sample(self,model,config,usable,target,linear,features,args,feature_sec,build_sec):
        if self.cut is None and len(usable)>target and any(f.is_frg_macro for f in features):
            self.clock.mark_boundary()
            state,start=boundary_state(args)
            differences=[k for k in sorted(set(state)|set(self.expected['state'])) if state.get(k)!=self.expected['state'].get(k)]
            if differences: raise RuntimeError('Prefix mismatch: '+json.dumps(differences[:30]))
            forced_indices(self.cell,len(usable),target)
            self.cut=dict(case=CASE,call=len(self.records),layer=args._layer_index,n=len(usable),k=target,
                          prefix_sec=self.clock.cutoff-start,requested_indices=list(CELLS[self.cell]))
            self.features=features;self.linear=linear
            self.clock.continue_live()
            if self.mode=='work':
                self.budget=WorkBudget(self.clock,args);self.budget.activate()
            self.original_repair=beam._repair_union_diversity_selection
            beam._repair_union_diversity_selection=self.forced_repair
            self.active=True
        # Every cell still receives the same native sample; only retention is forced.
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
            self.restore_repair();self.active=False


def freeze():
    if (OUT/'manifest.json').exists(): raise FileExistsError('Membership protocol already frozen')
    factorial_manifest()
    prefix=REPLAY/'prefixes'/f'{CASE}.json';clock=REPLAY/'prefixes'/f'{CASE}.clock.gz'
    data=json.loads(prefix.read_text(encoding='utf-8'))
    assert data['n']==48 and data['k']==4
    for index in (19,20,21,23): assert data['candidates'][index]['feature']['is_frg_macro']
    for index in (9,16): assert not data['candidates'][index]['feature']['is_frg_macro']
    sources=['scripts/production_route_membership.py','scripts/production_selector_factorial.py',
             'scripts/production_selector_replay.py','scripts/benchmark_compact_beam_qubo_n0.py','scripts/beam_selector_research.py']
    write_json(OUT/'manifest.json',dict(date='2026-10-02',case=CASE,cells=CELLS,common=[23,19],
        slot_a=dict(macro=20,ordinary=9),slot_b=dict(macro=21,ordinary=16),
        prefix=dict(path=prefix.relative_to(ROOT).as_posix(),sha256=digest(prefix)),
        clock=dict(path=clock.relative_to(ROOT).as_posix(),sha256=digest(clock)),
        source_sha256={p:digest(ROOT/p) for p in sources},work_settings=WORK_SETTINGS,
        successor_limit=SUCCESSOR_LIMIT,work_repeats=2,wall_repeats=3,watchdog_sec=120,
        reference_cells=dict(both='none',neither='quota_only'),
        intervention='Exact-k retention only; original repair skipped at that call and restored before later selectors; native output order retained.',
        objective='Original production factors retained; forced cells are not proxy optimizers.',
        scope='One C2 production boundary, not independent instances or default promotion',
        physical_time_scope='Solve includes reconstructed prefix and instrumentation; final validation follows. No work-mode logical time interpreted as seconds.'))


def manifest():
    data=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    factorial_manifest()
    for p,sha in data['source_sha256'].items():
        if digest(ROOT/p)!=sha: raise ValueError(f'Frozen source changed: {p}')
    for key in ('prefix','clock'):
        item=data[key]
        if digest(ROOT/item['path'])!=item['sha256']: raise ValueError('Frozen boundary changed')
    return data


def worker(cell,mode,repeat):
    m=manifest();data=next(c for c in business_manifest()['cases'] if c['case']==CASE)
    instance=load_instance(data['instance']);args=business_args()
    expected=json.loads((ROOT/m['prefix']['path']).read_text(encoding='utf-8'))
    with gzip.open(ROOT/m['clock']['path'],'rb') as stream:
        values=array('d');values.frombytes(stream.read())
    if expected['byteorder']!=sys.byteorder: values.byteswap()
    random.seed(37);np.random.seed(37)
    clock=SuffixClock(values,expected['cutoff'],mode)
    observer=MembershipObserver(clock,expected,cell,mode);args._union_selector_observer=observer
    start=REAL_TIME();time.perf_counter=clock
    try:
        result=beam.run_compact_beam_instance(CASE,instance,args)
    finally:
        time.perf_counter=REAL_TIME
        observer.restore_repair()
        if observer.budget is not None: observer.budget.restore()
    elapsed=REAL_TIME()-start
    if observer.cut is None: raise RuntimeError('Matched boundary was not reached')
    business_ok,goal_ok,message=beam._validate_instance_result(result,instance,business_alpha=1.3,business_cap_enabled=True)
    path=[asdict(move) for move in result.moves]
    work=dict(observer.budget.stats) if observer.budget is not None else {}
    telemetry={k:getattr(result,k) for k in ('expanded','generated','qubo_calls','post_polish_attempts',
        'post_polish_improvements','post_polish_moves_saved','post_polish_suffix_expanded',
        'post_polish_window_attempts','post_polish_bridge_expanded') if hasattr(result,k)}
    trace=[dict(call=r['call'],n=r['n'],k=r['k'],selected=r['selected']) for r in observer.records]
    signature=fingerprint(dict(path=path,work=work,telemetry=telemetry,selection_trace=trace))
    reference_match=None
    if mode=='work' and cell in m['reference_cells']:
        factor=m['reference_cells'][cell]
        reference=json.loads((ROUND6/'runs'/f'{CASE}__work__{factor}__r0.json').read_text(encoding='utf-8'))
        reference_match=signature==reference['row']['signature']
        if not reference_match: raise RuntimeError('Frozen endpoint control did not reproduce')
    row=dict(case=CASE,cell=cell,mode=mode,repeat=repeat,state_matched=True,
        valid=business_ok and goal_ok,business_ok=business_ok,goal_ok=goal_ok,moves=len(path) if goal_ok else None,
        selected=observer.target_record['selected'],macro_selected=observer.target_record['macro_selected'],
        original_proxy_objective=observer.target_record['original_proxy_objective'],
        physical_solver_sec=elapsed,charged_time_sec=result.runtime_sec if mode=='wall' else None,
        logical_within_budget=result.runtime_sec<=20 if mode=='wall' else None,
        reference_signature_matched=reference_match,signature=signature,validation_message=message,
        **telemetry,**work)
    write_json(OUT/'runs'/f'{mode}__{cell}__r{repeat}.json',dict(row=row,path=path,cut=observer.cut,
        target=observer.target_record,selection_trace=trace))
    print(json.dumps(row,ensure_ascii=False),flush=True)


def panel(mode):
    m=manifest();cells=list(CELLS)
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
                write_json(OUT/'runs'/f'{stem}.json',dict(row=dict(case=CASE,mode=mode,cell=cell,repeat=repeat,
                    valid=False,moves=None,state_matched=False,status='EXTERNAL_WATCHDOG'),path=[]))
                print(stem+' EXTERNAL_WATCHDOG',flush=True);continue
            if proc.returncode:
                write_json(OUT/'errors'/f'{stem}.json',dict(stdout=proc.stdout,stderr=proc.stderr,returncode=proc.returncode))
                raise RuntimeError(proc.stderr)
            print(proc.stdout.strip(),flush=True)
    rows=[json.loads(p.read_text(encoding='utf-8'))['row'] for p in sorted((OUT/'runs').glob('*.json'))]
    write_csv(OUT/'results.csv',rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run','worker'])
    parser.add_argument('--mode',choices=['work','wall'],default='work');parser.add_argument('--cell',choices=CELLS,default='both')
    parser.add_argument('--repeat',type=int,default=0);args=parser.parse_args()
    if args.command=='freeze': freeze()
    elif args.command=='run': panel(args.mode)
    else: worker(args.cell,args.mode,args.repeat)

if __name__=='__main__': main()
