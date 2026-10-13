"""Frozen opt-in cached-tail repair: matched continuations and full-start runs."""
from __future__ import annotations
import argparse
from array import array
from dataclasses import asdict
import gzip
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'scripts',ROOT/'src'): sys.path.insert(0,str(p))
import numpy as np
import benchmark_compact_beam_qubo_n0 as beam
from beam_selector_research import SelectorObserver
from frg_rollout_coordinates import CanonicalTailContract,replay_reference
from production_selector_replay import REAL_TIME,REPLAY,boundary_state,fingerprint,manifest as replay_manifest
from production_selector_factorial import SuffixClock,WorkBudget,WORK_SETTINGS,SUCCESSOR_LIMIT
from production_route_provenance import solver_frame
from probe_business_macro_windows import business_args,load_instance,manifest as business_manifest
from run_beam_selector_research import ENVIRONMENT,digest,write_json,write_csv

OUT=ROOT/'docs/experiments/beam_selector_round11_20261007'
ROUND10=ROOT/'docs/experiments/beam_selector_round10_20261007'
CASES=('A5_26_single','B3_26_single','C2_26_single','C4_26_single')
ARMS=('legacy','fixed')
TELEMETRY=('expanded','generated','qubo_calls','post_polish_attempts','post_polish_improvements',
    'post_polish_moves_saved','post_polish_suffix_expanded','post_polish_window_attempts','post_polish_bridge_expanded')


def cache_audit(args):
    """Strict child-start physical replay, including partial records; no search."""
    rows=[]
    for code,record in sorted(getattr(args,'_union_rollout_probe_cache',{}).items()):
        row=dict(code=code.hex(),feasible=record.feasible,final_n=record.final_n,
            moves=record.moves,path_length=len(record.move_path),status=record.status,
            local_steps=[m.step for m in record.move_path]==list(range(len(record.move_path))),
            length_matches=record.moves==len(record.move_path))
        try:
            _,endpoint,n=replay_reference(beam._code_to_stacks(code,args._stack_count,args._max_height),args._max_height,record.move_path)
            if n!=record.final_n or (record.feasible and n!=0): raise ValueError('Tail endpoint disagrees with metadata')
            row.update(physical_ok=True,endpoint=endpoint.hex(),actual_n=n,message='ok')
        except (KeyError,ValueError,RuntimeError,IndexError) as error:
            row.update(physical_ok=False,message=str(error))
        row['contract_ok']=row['physical_ok'] and row['local_steps'] and row['length_matches']
        rows.append(row)
    return dict(entries=len(rows),physical_invalid=sum(not r['physical_ok'] for r in rows),
        complete_claim_invalid=sum(r['feasible'] and not r['physical_ok'] for r in rows),
        nonlocal_steps=sum(not r['local_steps'] for r in rows),length_mismatch=sum(not r['length_matches'] for r in rows),
        contract_invalid=sum(not r['contract_ok'] for r in rows),records=rows)


class ContractObserver(SelectorObserver):
    def __init__(self,arm,clock=None,expected=None):
        super().__init__('sa',capture_limit=0)
        self.contract=CanonicalTailContract();self.contract_arm=arm;self.clock=clock;self.expected=expected
        self.cut=None;self.cut_active=False;self.target_record=None;self.budget=None;self.migration=[];self.migration_sec=0.

    def sample(self,model,config,usable,target,linear,features,args,feature_sec,build_sec):
        if self.clock is not None and self.cut is None and len(usable)>target and any(f.is_frg_macro for f in features):
            self.clock.mark_boundary()
            state,start=boundary_state(args)
            mismatch=[k for k in sorted(set(state)|set(self.expected['state'])) if state.get(k)!=self.expected['state'].get(k)]
            if mismatch: raise RuntimeError('Prefix mismatch: '+json.dumps(mismatch[:30]))
            self.cut=dict(layer=args._layer_index,call=len(self.records),n=len(usable),k=target,
                prefix_sec=self.clock.cutoff-start,state_signature=fingerprint(state),state_matched=True)
            self.cut_active=True;self.clock.continue_live()
            if self.clock.mode=='work':
                self.budget=WorkBudget(self.clock,args);self.budget.activate()
            if self.contract_arm=='fixed':
                started=REAL_TIME();self.contract.install()
                caller=inspect.currentframe().f_back
                while caller is not None and caller.f_code.co_name!='_select_union_diversity_qubo_supplement': caller=caller.f_back
                if caller is None: raise RuntimeError('Native union selector frame missing')
                self.migration=self.contract.migrate_current(args,solver_frame().f_locals['candidates'],caller.f_locals['rollout_by_code'])
                self.migration_sec=REAL_TIME()-started
        return super().sample(model,config,usable,target,linear,features,args,feature_sec,build_sec)

    def finish(self,ordered,selector_sec):
        super().finish(ordered,selector_sec)
        if self.cut_active:
            self.target_record=dict(self.current);self.cut_active=False

    def restore(self):
        self.contract.restore()
        if self.budget is not None: self.budget.restore()


def freeze():
    if (OUT/'manifest.json').exists(): raise FileExistsError('Round 11 already frozen')
    replay_manifest();bm=business_manifest()
    sources=['scripts/run_rollout_contract.py','scripts/frg_rollout_coordinates.py','scripts/benchmark_compact_beam_qubo_n0.py',
        'scripts/beam_selector_research.py','scripts/production_selector_replay.py','scripts/production_selector_factorial.py',
        'scripts/production_route_provenance.py','scripts/probe_business_macro_windows.py','scripts/run_beam_selector_research.py']
    inputs=[]
    for case in CASES:
        for path in [REPLAY/'prefixes'/f'{case}.json',REPLAY/'prefixes'/f'{case}.clock.gz',
            REPLAY/'results'/f'{case}__capture.json',ROUND10/'runs'/f'{case}__work__native_audited__r0.json']:
            inputs.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=digest(path)))
    write_json(OUT/'manifest.json',dict(date='2026-10-07',cases=CASES,arms=ARMS,
        source_sha256={p:digest(ROOT/p) for p in sources},inputs=inputs,
        instances={c['case']:c['instance'] for c in bm['cases'] if c['case'] in CASES},
        work_settings=WORK_SETTINGS,successor_limit=SUCCESSOR_LIMIT,matched_work_repeats=2,matched_wall_repeats=3,
        full_wall_repeats=3,planned_runs=64,watchdog_sec=120,request_sec=20.,seed=37,alpha=1.3,
        environment=ENVIRONMENT,python=sys.version,numpy=np.__version__,
        contract='Child-start tail, canonical child stack indices, local step numbers, moves=len(tail).',
        producers='Frontier permutation; macro-chain/unified executed-prefix removal before metadata comparison; ordinary probe local steps.',
        consumer='Expand all macro ancestry, lift physical prefix, translate canonical tail, renumber and strictly validate goal/business cap.',
        migration='After exact boundary match: replay all cache entries; repair current macro origins with verified parent/prefix; invalidate unverified entries. Subsequent recomputation charged.',
        matched_scope='First already-built selector stays native. Install only after clock/frame/args/cache/RNG match. Later search and post-polish use selected contract.',
        full_scope='Install fixed writers before initialization. Native real clock from original instance; no shared-checkpoint claim.',
        time_scope='Report measured physical solve, validation/cache audit and their sum. Matched logical timer separately; original prefix runtime differs from physical replay.',
        gates='All 16 work rows: exact prefix and first-choice match, exact repeated path/work/selection signatures; legacy matches round10 native control. Retain all wall failures and regressions. Corrected cache invalidity fails correctness gate.',
        claims='Correctness prerequisite only; no selector objective change, novelty, hardware advantage or default promotion.'))


def manifest():
    m=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'));business_manifest();replay_manifest()
    for p,sha in m['source_sha256'].items():
        if digest(ROOT/p)!=sha: raise ValueError('Frozen source changed: '+p)
    for item in m['inputs']:
        if digest(ROOT/item['path'])!=item['sha256']: raise ValueError('Frozen input changed: '+item['path'])
    return m


def stem(case,stage,mode,arm,repeat): return f'{case}__{stage}__{mode}__{arm}__r{repeat}'


def worker(case,stage,mode,arm,repeat):
    m=manifest();instance=load_instance(m['instances'][case]);args=business_args()
    random.seed(37);np.random.seed(37)
    clock=None;expected=None
    if stage=='matched':
        expected=json.loads((REPLAY/'prefixes'/f'{case}.json').read_text(encoding='utf-8'))
        with gzip.open(REPLAY/'prefixes'/f'{case}.clock.gz','rb') as stream:
            values=array('d');values.frombytes(stream.read())
        if expected['byteorder']!=sys.byteorder: values.byteswap()
        clock=SuffixClock(values,expected['cutoff'],mode)
    elif mode!='wall': raise ValueError('Full-start stage uses a native wall clock')
    observer=ContractObserver(arm,clock,expected);args._union_selector_observer=observer
    started=REAL_TIME()
    try:
        if clock is not None: time.perf_counter=clock
        elif arm=='fixed': observer.contract.install()
        result=beam.run_compact_beam_instance(case,instance,args)
    finally:
        time.perf_counter=REAL_TIME;observer.restore()
    physical=REAL_TIME()-started
    audit_started=REAL_TIME()
    business_ok,goal_ok,message=beam._validate_instance_result(result,instance,business_alpha=1.3,business_cap_enabled=True)
    audit=cache_audit(args);audit_sec=REAL_TIME()-audit_started
    path=[asdict(move) for move in result.moves]
    work=dict(observer.budget.stats) if observer.budget is not None else {}
    telemetry={k:getattr(result,k) for k in TELEMETRY if hasattr(result,k)}
    trace=[dict(call=r['call'],n=r['n'],k=r['k'],selected=r['selected']) for r in observer.records]
    signature=fingerprint(dict(path=path,work=work,telemetry=telemetry,selection_trace=trace))
    first_equal=None;old_equal=None
    if stage=='matched':
        if observer.cut is None or observer.target_record is None: raise RuntimeError('Matched boundary missing')
        capture=json.loads((REPLAY/'results'/f'{case}__capture.json').read_text(encoding='utf-8'))
        first_equal=observer.target_record['selected']==capture['target_record']['selected']
        if mode=='work' and arm=='legacy':
            ref=json.loads((ROUND10/'runs'/f'{case}__work__native_audited__r0.json').read_text(encoding='utf-8'))
            old_equal=signature==ref['row']['signature']
    row=dict(case=case,stage=stage,mode=mode,arm=arm,repeat=repeat,
        state_matched=observer.cut is not None if stage=='matched' else None,first_choice_matched=first_equal,
        native_reference_matched=old_equal,valid=business_ok and goal_ok,business_ok=business_ok,goal_ok=goal_ok,
        moves=len(path) if goal_ok else None,status=result.status,validation_message=message,
        physical_solver_sec=physical,validation_cache_audit_sec=audit_sec,measured_end_to_end_sec=physical+audit_sec,
        charged_time_sec=result.runtime_sec if mode=='wall' else None,logical_within_budget=result.runtime_sec<=20 if mode=='wall' else None,
        physical_within_budget=physical<=20.,measured_end_to_end_within_budget=physical+audit_sec<=20.,
        signature=signature,cache_entries=audit['entries'],cache_physical_invalid=audit['physical_invalid'],
        cache_complete_claim_invalid=audit['complete_claim_invalid'],cache_nonlocal_steps=audit['nonlocal_steps'],
        cache_contract_invalid=audit['contract_invalid'],migration_sec=observer.migration_sec,
        migration_entries=len(observer.migration),migration_corrected=sum(r['status']=='CANONICAL_CHILD_TAIL' for r in observer.migration),
        migration_invalidated=sum(r['status']=='INVALIDATED_UNVERIFIED' for r in observer.migration),
        contract_stats=observer.contract.stats,**telemetry,**work)
    write_json(OUT/'runs'/f'{stem(case,stage,mode,arm,repeat)}.json',dict(row=row,path=path,cut=observer.cut,
        first_selector=observer.target_record,selection_trace=trace,migration=observer.migration,
        producer_records=observer.contract.records,cache_audit=audit))
    if stage=='matched' and not first_equal: raise RuntimeError('First already-built native choice changed')
    if old_equal is False: raise RuntimeError('Legacy work signature differs from round10 native control')
    if stage=='matched' and mode=='work' and repeat:
        ref=json.loads((OUT/'runs'/f'{stem(case,stage,mode,arm,0)}.json').read_text(encoding='utf-8'))
        if signature!=ref['row']['signature']: raise RuntimeError('Work repetition differs')
    print(json.dumps({k:row[k] for k in ('case','stage','mode','arm','repeat','valid','moves','physical_solver_sec',
        'cache_entries','cache_physical_invalid','cache_contract_invalid','migration_invalidated','native_reference_matched')},ensure_ascii=False),flush=True)


def audit_work():
    rows=[]
    for case in CASES:
        for arm in ARMS:
            pair=[json.loads((OUT/'runs'/f'{stem(case,"matched","work",arm,r)}.json').read_text(encoding='utf-8'))['row'] for r in range(2)]
            if not all(r['state_matched'] and r['first_choice_matched'] for r in pair) or pair[0]['signature']!=pair[1]['signature']:
                raise RuntimeError('Work/state gate failed: '+case+' '+arm)
            if arm=='legacy' and not all(r['native_reference_matched'] for r in pair): raise RuntimeError('Legacy control gate failed')
            if arm=='fixed' and any(r['cache_contract_invalid'] for r in pair): raise RuntimeError('Repaired cache contract failed')
            rows.append(dict(case=case,arm=arm,repeat_equal=True,signature=pair[0]['signature'],legacy_control=pair[0]['native_reference_matched']))
    write_json(OUT/'work_repeat_audit.json',rows)
    return rows


def panel(stage,mode):
    m=manifest()
    if stage=='full' and mode!='wall': raise ValueError('Full-start stage uses wall mode')
    if stage=='full' or mode=='wall': audit_work()
    for repeat in range(m[f'{stage}_{mode}_repeats']):
        for ci,case in enumerate(CASES):
            order=ARMS if (ci+repeat)%2==0 else ARMS[::-1]
            for arm in order:
                name=stem(case,stage,mode,arm,repeat)
                if (OUT/'runs'/f'{name}.json').exists(): continue
                command=[sys.executable,str(Path(__file__).resolve()),'worker','--case',case,'--stage',stage,
                    '--mode',mode,'--arm',arm,'--repeat',str(repeat)]
                try:
                    proc=subprocess.run(command,cwd=ROOT,env={**os.environ,**ENVIRONMENT},capture_output=True,
                        text=True,encoding='utf-8',errors='replace',timeout=m['watchdog_sec'])
                except subprocess.TimeoutExpired as error:
                    write_json(OUT/'runs'/f'{name}.json',dict(row=dict(case=case,stage=stage,mode=mode,arm=arm,
                        repeat=repeat,valid=False,moves=None,status='EXTERNAL_WATCHDOG'),path=[]))
                    print(name+' EXTERNAL_WATCHDOG',flush=True);continue
                if proc.returncode:
                    p=OUT/'errors'/f'{name}.json';suffix=1
                    while p.exists(): p=OUT/'errors'/f'{name}__error{suffix}.json';suffix+=1
                    write_json(p,dict(stdout=proc.stdout,stderr=proc.stderr,returncode=proc.returncode))
                    raise RuntimeError(proc.stderr)
                print(proc.stdout.strip(),flush=True)
    rows=[json.loads(p.read_text(encoding='utf-8'))['row'] for p in sorted((OUT/'runs').glob('*.json'))]
    write_csv(OUT/'results.csv',rows)
    if stage=='matched' and mode=='work': audit_work()


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run','worker','audit-work'])
    p.add_argument('--case',choices=CASES,default=CASES[0]);p.add_argument('--arm',choices=ARMS,default='legacy')
    p.add_argument('--stage',choices=['matched','full'],default='matched');p.add_argument('--mode',choices=['work','wall'],default='work')
    p.add_argument('--repeat',type=int,default=0);a=p.parse_args()
    if a.command=='freeze': freeze()
    elif a.command=='run': panel(a.stage,a.mode)
    elif a.command=='audit-work': audit_work()
    else: worker(a.case,a.stage,a.mode,a.arm,a.repeat)

if __name__=='__main__': main()
