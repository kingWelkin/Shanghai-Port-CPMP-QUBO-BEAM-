"""One native selector ablation on canonical cached tails and matched state."""
from __future__ import annotations
import argparse
from array import array
import copy
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
from beam_selector_research import SelectorObserver,encode,restore,decode_feature
from container_reshuffle.qubo.model import QuboModel
from frg_rollout_coordinates import CanonicalTailContract
from production_selector_replay import REAL_TIME,boundary_state,fingerprint
from production_selector_factorial import SuffixClock,WorkBudget,WORK_SETTINGS,SUCCESSOR_LIMIT,set_factors,restore_factors
from production_route_provenance import solver_frame
from probe_business_macro_windows import business_args,load_instance
from run_rollout_contract import manifest as round11_manifest,cache_audit,TELEMETRY
from run_beam_selector_research import ENVIRONMENT,digest,write_json,write_csv

OUT=ROOT/'docs/experiments/beam_selector_round12_20261007'
CASES=('A5_26_single','B3_26_single','C2_26_single','C4_26_single')
ARMS={'linear':('none','linear'),'pair_greedy':('pair_only','marginal'),
    'quota_greedy':('quota_only','marginal'),'full_greedy':('full','marginal'),'full_sa':('full','sa')}


def scalar_settings(args):
    settings={k:v for k,v in vars(args).items() if isinstance(v,(str,int,float,bool,type(None)))}
    settings['_quality_reservoir_objective_telemetry_enabled']=False
    return settings


def variant_model(packet,factor):
    """Compile the native exact-k objective with fixed q(k), for offline audit."""
    linear,features,args=restore(packet);set_factors(args,factor)
    model=QuboModel();variables=[model.add_binary_var(f'ud_{i}') for i in range(len(linear))]
    for i,v in enumerate(linear): model.add_linear(variables[i],v)
    pair_args=copy.copy(args);pair_args.union_diversity_qubo_adaptive_macro_quota_enabled=False
    for i in range(len(linear)):
        for j in range(i+1,len(linear)):
            value=beam._union_diversity_objective({i,j},linear,features,pair_args)-linear[i]-linear[j]
            if abs(value)>1e-12: model.add_quadratic(variables[i],variables[j],value)
    beam._add_union_diversity_adaptive_macro_quota_terms(model,variables,features,packet['target'],args)
    model.add_exactly_k_penalty(variables,packet['target'],float(getattr(args,'qubo_penalty_a',30.) or 30.))
    quota=beam._union_diversity_adaptive_macro_quota_target(features,packet['target'],args)
    quota_constant=float(getattr(args,'union_diversity_qubo_macro_quota_penalty',0.) or 0.)*quota*quota
    return model,args,quota_constant


class CorrectedObserver(SelectorObserver):
    def __init__(self,case,clock,contract,arm='full_sa',mode='wall',expected=None):
        super().__init__('sa',capture_limit=0)
        self.case=case;self.clock=clock;self.contract=contract;self.branch=arm;self.mode=mode;self.expected=expected
        self.cut=None;self.cut_active=False;self.target_record=None;self.packet=None;self.budget=None
        self.saved=None;self.changed_args=None;self.originals={};self.commit_pending=False;self.frontier_verified=False
        self.commit_record=None;self.actual_frontier=None

    def sample(self,model,config,usable,target,linear,features,args,feature_sec,build_sec):
        if self.cut is None and len(usable)>target and any(beam._union_diversity_is_macro_feature(f) for f in features):
            self.clock.mark_boundary();state,start=boundary_state(args)
            stats_signature=fingerprint(self.contract.stats)
            if self.expected is not None:
                mismatch=[k for k in sorted(set(state)|set(self.expected['state'])) if state.get(k)!=self.expected['state'].get(k)]
                if mismatch: raise RuntimeError('Corrected prefix mismatch: '+json.dumps(mismatch[:30]))
                if stats_signature!=self.expected['contract_stats_signature']: raise RuntimeError('Producer/graft prefix state differs')
            self.clock.continue_live()
            caller=inspect.currentframe().f_back
            rollouts=caller.f_locals['rollout_by_code']
            packet=dict(schema='corrected-native-selector/v1',linear=list(linear),features=encode([f._asdict() for f in features]),
                settings=scalar_settings(args),model=model.to_dict(),sampler_config=asdict(config),target=target,
                usable=encode(usable),core=[c.hex() for c in self.layer_context[1]],
                rollout_by_code={code.hex():asdict(r) for code,r in sorted(rollouts.items())})
            self.packet=json.loads(json.dumps(packet,ensure_ascii=False))
            packet_signature=fingerprint(self.packet)
            if self.expected is not None and packet_signature!=self.expected['packet_signature']: raise RuntimeError('Shared corrected evidence packet differs')
            self.cut=dict(case=self.case,layer=args._layer_index,call=len(self.records),n=len(usable),k=target,
                macro_count=sum(beam._union_diversity_is_macro_feature(f) for f in features),
                clock_calls=len(self.clock.values),cutoff=self.clock.cutoff,prefix_sec=self.clock.cutoff-start,
                state=state,state_signature=fingerprint(state),contract_stats_signature=stats_signature,
                packet_signature=packet_signature,byteorder=sys.byteorder)
            self.features=features;self.linear=linear;self.usable=list(usable);self.protected=set(self.layer_context[1])
            if self.mode=='work': self.budget=WorkBudget(self.clock,args);self.budget.activate()
            if self.expected is not None:
                factor,sampling=ARMS[self.branch]
                self.saved=set_factors(args,factor);self.changed_args=args;self.arm=sampling
            for name,callback in {'_record_quality_reservoir_selected_telemetry':self.commit,
                '_select_next_frontier':self.check_next,'CompactBeamResult':self.result}.items():
                self.originals[name]=getattr(beam,name);setattr(beam,name,callback)
            self.cut_active=True;self.commit_pending=True
        return super().sample(model,config,usable,target,linear,features,args,feature_sec,build_sec)

    def finish(self,ordered,selector_sec):
        super().finish(ordered,selector_sec)
        if self.cut_active:
            args=inspect.currentframe().f_back.f_locals['args'];shadow=copy.copy(args)
            shadow._quality_reservoir_objective_telemetry_enabled=False
            selected=set(ordered);unary=sum(self.linear[i] for i in selected)
            quota=beam._union_diversity_adaptive_macro_quota_penalty(selected,self.features,shadow)
            value=beam._union_diversity_objective(selected,self.linear,self.features,shadow)
            self.target_record=dict(self.current,selected_codes=[self.usable[i][0].hex() for i in ordered],
                macro_selected=sum(beam._union_diversity_is_macro_feature(self.features[i]) for i in selected),
                unary=unary,quota=quota,pair=value-unary-quota,objective=value,
                quota_target=beam._union_diversity_adaptive_macro_quota_target(self.features,self.current['k'],shadow))
            if self.saved is not None: restore_factors(args,self.saved);self.saved=None
            self.arm='sa';self.cut_active=False

    def commit(self,selected_codes,args):
        if self.commit_pending:
            loc=solver_frame().f_locals
            if loc['layer_index']!=self.cut['layer']: raise RuntimeError('Filled Beam commit moved to another layer')
            if not self.protected<=set(selected_codes): raise RuntimeError('Protected core lost at native fill')
            candidates=loc['candidates']
            self.expected_frontier=[(c,candidates[c][0],candidates[c][1]) for c in selected_codes]
            self.commit_record=dict(layer=loc['layer_index'],width=len(selected_codes),
                codes=[c.hex() for c in selected_codes],protected=[c.hex() for c in sorted(self.protected)],
                source_by_code={c.hex():getattr(args,'_frontier_trace_last_selected_sources',{}).get(c) for c in selected_codes})
            self.commit_pending=False
            beam._record_quality_reservoir_selected_telemetry=self.originals['_record_quality_reservoir_selected_telemetry']
        return self.originals['_record_quality_reservoir_selected_telemetry'](selected_codes,args)

    def verify_frontier(self,actual):
        if self.commit_pending: raise RuntimeError('Frontier observed before filled Beam commitment')
        if self.expected_frontier!=actual: raise RuntimeError('Actual native frontier differs from filled Beam')
        self.actual_frontier=[[c.hex(),n,d] for c,n,d in actual];self.frontier_verified=True

    def check_next(self,candidates,args):
        if not self.frontier_verified: self.verify_frontier(solver_frame().f_locals['frontier'])
        beam._select_next_frontier=self.originals['_select_next_frontier']
        return self.originals['_select_next_frontier'](candidates,args)

    def result(self,*args,**kwargs):
        if not self.frontier_verified: self.verify_frontier(solver_frame().f_locals['frontier'])
        return self.originals['CompactBeamResult'](*args,**kwargs)

    def restore(self):
        if self.saved is not None: restore_factors(self.changed_args,self.saved);self.saved=None
        for name,original in self.originals.items(): setattr(beam,name,original)
        if self.budget is not None: self.budget.restore()


def freeze_capture():
    if (OUT/'capture_manifest.json').exists(): raise FileExistsError('Capture protocol already frozen')
    previous=round11_manifest()
    sources=['scripts/production_corrected_attribution.py','scripts/frg_rollout_coordinates.py',
        'scripts/benchmark_compact_beam_qubo_n0.py','scripts/beam_selector_research.py','scripts/production_selector_replay.py',
        'scripts/production_selector_factorial.py','scripts/production_route_provenance.py','scripts/run_rollout_contract.py',
        'scripts/probe_business_macro_windows.py','scripts/run_beam_selector_research.py']
    write_json(OUT/'capture_manifest.json',dict(date='2026-10-07',cases=CASES,arms=ARMS,
        instances=previous['instances'],source_sha256={p:digest(ROOT/p) for p in sources},
        seed=37,request_sec=20.,alpha=1.3,work_settings=WORK_SETTINGS,successor_limit=SUCCESSOR_LIMIT,
        work_repeats=2,wall_repeats=3,watchdog_sec=120,total_runs_including_capture=104,environment=ENVIRONMENT,
        rule='First corrected native union decision with n>k and at least one macro feature, all four predeclared cases.',
        interventions='One initial selection and its native repair/sort; restore pair/quota flags immediately after that decision. Later native decisions identical policy.',
        common_evidence='Canonical-tail contract from initialization in every run; exact clock/native state/cache/RNG, contract counters and packet match; no additional probes.',
        actual_beam='Record native filled Beam after deterministic fill, confirm identical ordered list is used as actual next frontier. Report membership and order separately.',
        gating='40 work rows must match prefix and repeat path/work/selection/actual-frontier signatures; full_sa must reproduce captured first choice and filled Beam. Cache contract must hold.',
        time='Historical prefix clock plus live suffix, 20-second nominal request; physical recomputed prefix, validation/cache audit separately measured. No fresh full-start claim for matched branches.',
        proxy='Offline exhaustive exact-k audit only when n choose k <= 300000; never a terminal quality certificate.',
        scope='Four development instances, one seed and one varied decision; no novel feature, weight tuning, held-out generalization or default promotion.'))


def capture_manifest():
    m=json.loads((OUT/'capture_manifest.json').read_text(encoding='utf-8'));round11_manifest()
    for p,sha in m['source_sha256'].items():
        if digest(ROOT/p)!=sha: raise ValueError('Frozen capture source changed: '+p)
    return m


def freeze_panel():
    if (OUT/'manifest.json').exists(): raise FileExistsError('Continuation protocol already frozen')
    m=capture_manifest();inputs=[]
    for case in CASES:
        for path in [OUT/'prefixes'/f'{case}.json',OUT/'prefixes'/f'{case}.clock.gz',OUT/'captures'/f'{case}.json']:
            if not path.exists(): raise FileNotFoundError('Predeclared capture missing: '+case)
            inputs.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=digest(path)))
    write_json(OUT/'manifest.json',dict(capture_manifest_sha256=digest(OUT/'capture_manifest.json'),inputs=inputs,
        cases=CASES,arms=ARMS,work_repeats=m['work_repeats'],wall_repeats=m['wall_repeats'],watchdog_sec=m['watchdog_sec']))


def manifest():
    m=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'));capture_manifest()
    if digest(OUT/'capture_manifest.json')!=m['capture_manifest_sha256']: raise ValueError('Capture manifest changed')
    for item in m['inputs']:
        if digest(ROOT/item['path'])!=item['sha256']: raise ValueError('Frozen corrected prefix input changed: '+item['path'])
    return m


def worker(case,arm,mode,repeat,capture=False):
    m=capture_manifest() if capture else {**capture_manifest(),**manifest()}
    instance=load_instance(m['instances'][case]);args=business_args();contract=CanonicalTailContract()
    expected=None;values=None;cutoff=None
    if not capture:
        expected=json.loads((OUT/'prefixes'/f'{case}.json').read_text(encoding='utf-8'))
        with gzip.open(OUT/'prefixes'/f'{case}.clock.gz','rb') as s: values=array('d');values.frombytes(s.read())
        if expected['byteorder']!=sys.byteorder: values.byteswap()
        cutoff=expected['cutoff']
    clock=SuffixClock(values,cutoff,mode);observer=CorrectedObserver(case,clock,contract,arm,mode,expected)
    args._union_selector_observer=observer;random.seed(37);np.random.seed(37)
    started=REAL_TIME()
    try:
        contract.install();time.perf_counter=clock
        result=beam.run_compact_beam_instance(case,instance,args)
    finally:
        time.perf_counter=REAL_TIME;observer.restore();contract.restore()
    physical=REAL_TIME()-started
    if observer.cut is None or not observer.frontier_verified: raise RuntimeError('No predeclared boundary/verified actual frontier')
    audited=REAL_TIME();business_ok,goal_ok,message=beam._validate_instance_result(result,instance,business_alpha=1.3,business_cap_enabled=True)
    audit=cache_audit(args);audit_sec=REAL_TIME()-audited
    path=[asdict(move) for move in result.moves];work=dict(observer.budget.stats) if observer.budget is not None else {}
    telemetry={k:getattr(result,k) for k in TELEMETRY if hasattr(result,k)}
    trace=[dict(call=r['call'],n=r['n'],k=r['k'],selected=r['selected']) for r in observer.records]
    signature=fingerprint(dict(path=path,work=work,telemetry=telemetry,selection_trace=trace,actual_frontier=observer.actual_frontier))
    baseline=None;choice_equal=None;actual_changed=None;order_changed=None;added=[];removed=[]
    if not capture:
        baseline=json.loads((OUT/'captures'/f'{case}.json').read_text(encoding='utf-8'))
        choice_equal=observer.target_record['selected']==baseline['target_record']['selected']
        old=set(baseline['filled_beam']['codes']);new=set(observer.commit_record['codes'])
        added=sorted(new-old);removed=sorted(old-new);actual_changed=bool(added or removed)
        order_changed=observer.commit_record['codes']!=baseline['filled_beam']['codes']
    row=dict(case=case,arm=arm,mode=mode,repeat=repeat,capture=capture,state_matched=not capture,
        actual_frontier_verified=True,valid=business_ok and goal_ok,business_ok=business_ok,goal_ok=goal_ok,
        moves=len(path) if goal_ok else None,status=result.status,validation_message=message,
        n=observer.cut['n'],k=observer.cut['k'],layer=observer.cut['layer'],selected=observer.target_record['selected'],
        macro_selected=observer.target_record['macro_selected'],objective=observer.target_record['objective'],
        unary=observer.target_record['unary'],pair=observer.target_record['pair'],quota=observer.target_record['quota'],
        raw_count_error=observer.target_record['raw_count_error'],
        filled_width=observer.commit_record['width'],actual_membership_changed=actual_changed,actual_order_changed=order_changed,
        actual_added=len(added),actual_removed=len(removed),native_choice_matched=choice_equal,
        packet_signature=observer.cut['packet_signature'],signature=signature,
        physical_solver_sec=physical,validation_cache_audit_sec=audit_sec,measured_end_to_end_sec=physical+audit_sec,
        charged_time_sec=result.runtime_sec if mode=='wall' else None,logical_within_budget=result.runtime_sec<=20 if mode=='wall' else None,
        measured_end_to_end_within_budget=physical+audit_sec<=20.,
        cache_entries=audit['entries'],cache_contract_invalid=audit['contract_invalid'],
        contract_stats=contract.stats,**telemetry,**work)
    data=dict(row=row,path=path,target_record=observer.target_record,filled_beam=observer.commit_record,
        actual_frontier=observer.actual_frontier,actual_added_codes=added,actual_removed_codes=removed,
        selection_trace=trace,cache_audit=audit)
    if capture:
        write_json(OUT/'prefixes'/f'{case}.json',dict(observer.cut,packet=observer.packet))
        p=OUT/'prefixes'/f'{case}.clock.gz';p.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(p,'wb') as s:s.write(clock.values.tobytes())
        write_json(OUT/'captures'/f'{case}.json',data)
    else:
        write_json(OUT/'runs'/f'{case}__{mode}__{arm}__r{repeat}.json',data)
        if arm=='full_sa' and (not choice_equal or actual_changed or order_changed): raise RuntimeError('Native corrected first choice/Beam did not reproduce')
        if mode=='work' and repeat:
            old=json.loads((OUT/'runs'/f'{case}__work__{arm}__r0.json').read_text(encoding='utf-8'))
            if signature!=old['row']['signature']: raise RuntimeError('Corrected work repetition differs')
    if audit['contract_invalid']: raise RuntimeError('Corrected cache contract violated')
    print(json.dumps({k:row[k] for k in ('case','arm','mode','repeat','capture','valid','moves','selected',
        'actual_added','actual_removed','actual_order_changed','physical_solver_sec','cache_contract_invalid')},ensure_ascii=False),flush=True)


def audit_work():
    rows=[]
    for case in CASES:
        signatures=set()
        for arm in ARMS:
            pair=[json.loads((OUT/'runs'/f'{case}__work__{arm}__r{r}.json').read_text(encoding='utf-8'))['row'] for r in range(2)]
            if not all(r['state_matched'] and r['actual_frontier_verified'] and not r['cache_contract_invalid'] for r in pair): raise RuntimeError('Work state/contract gate failed')
            if pair[0]['signature']!=pair[1]['signature']: raise RuntimeError('Work repeat gate failed')
            if arm=='full_sa' and not all(r['native_choice_matched'] and not r['actual_order_changed'] for r in pair): raise RuntimeError('Native corrected control gate failed')
            signatures.update(r['packet_signature'] for r in pair)
            rows.append(dict(case=case,arm=arm,repeat_equal=True,signature=pair[0]['signature']))
        if len(signatures)!=1: raise RuntimeError('Corrected common packet gate failed')
    write_json(OUT/'work_repeat_audit.json',rows);return rows


def panel(mode,capture=False):
    m=capture_manifest() if capture else manifest()
    if not capture and mode=='wall': audit_work()
    repeats=1 if capture else m[mode+'_repeats']
    arms=['full_sa'] if capture else list(ARMS)
    for repeat in range(repeats):
        for ci,case in enumerate(CASES):
            offset=(ci+repeat)%len(arms);order=arms[offset:]+arms[:offset]
            for arm in order:
                name=case if capture else f'{case}__{mode}__{arm}__r{repeat}'
                p=OUT/('captures' if capture else 'runs')/f'{name}.json'
                if p.exists(): continue
                command=[sys.executable,str(Path(__file__).resolve()),'worker','--case',case,'--arm',arm,'--mode',mode,'--repeat',str(repeat)]
                if capture: command.append('--capture')
                try:
                    proc=subprocess.run(command,cwd=ROOT,env={**os.environ,**ENVIRONMENT},capture_output=True,
                        text=True,encoding='utf-8',errors='replace',timeout=m['watchdog_sec'])
                except subprocess.TimeoutExpired as error:
                    write_json(p,dict(row=dict(case=case,arm=arm,mode=mode,repeat=repeat,capture=capture,
                        valid=False,moves=None,status='EXTERNAL_WATCHDOG'),path=[]))
                    print(name+' EXTERNAL_WATCHDOG',flush=True);continue
                if proc.returncode:
                    q=OUT/'errors'/f'{name}.json';suffix=1
                    while q.exists(): q=OUT/'errors'/f'{name}__error{suffix}.json';suffix+=1
                    write_json(q,dict(stdout=proc.stdout,stderr=proc.stderr,returncode=proc.returncode))
                    raise RuntimeError(proc.stderr)
                print(proc.stdout.strip(),flush=True)
    if not capture:
        rows=[json.loads(p.read_text(encoding='utf-8'))['row'] for p in sorted((OUT/'runs').glob('*.json'))]
        write_csv(OUT/'results.csv',rows)
        if mode=='work': audit_work()


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze-capture','capture','freeze-panel','run','worker','audit-work'])
    p.add_argument('--case',choices=CASES,default=CASES[0]);p.add_argument('--arm',choices=list(ARMS),default='full_sa')
    p.add_argument('--mode',choices=['work','wall'],default='wall');p.add_argument('--repeat',type=int,default=0);p.add_argument('--capture',action='store_true')
    a=p.parse_args()
    if a.command=='freeze-capture': freeze_capture()
    elif a.command=='capture': panel('wall',True)
    elif a.command=='freeze-panel': freeze_panel()
    elif a.command=='run': panel(a.mode)
    elif a.command=='audit-work': audit_work()
    else: worker(a.case,a.arm,a.mode,a.repeat,a.capture)

if __name__=='__main__': main()
