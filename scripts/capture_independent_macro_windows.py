"""Frozen independent CPMP instances and identifiable post-fill macro windows."""
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
for p in (ROOT,ROOT/'scripts',ROOT/'src'):sys.path.insert(0,str(p))
import numpy as np
import benchmark_compact_beam_qubo_n0 as beam
from beam_selector_research import SelectorObserver,encode
from cpmp_budget_witness import replay_prefix
from frg_rollout_coordinates import CanonicalTailContract
from production_selector_replay import PrefixClock,REAL_TIME,fingerprint
from production_route_provenance import solver_frame
from production_corrected_attribution import scalar_settings,manifest as previous_manifest
from probe_business_macro_windows import business_args,load_instance
from run_rollout_contract import cache_audit,TELEMETRY
from run_beam_selector_research import ENVIRONMENT,digest,write_json,write_csv

OUT=ROOT/'docs/experiments/beam_selector_round15_20261007'
TEMPLATES={'balanced':[4]*10,'skewed':[6,6,6,6,6,5,5,0,0,0]}
MIN_POOL=4;MAX_POOL=4


def independent_instances():
    cases=[];seen=set()
    for ti,(template,heights) in enumerate(TEMPLATES.items()):
        for i in range(10):
            seed=2026100700+1000*ti+i;values=list(range(1,41));random.Random(seed).shuffle(values)
            stacks=[];offset=0
            for h in heights:stacks.append(values[offset:offset+h]);offset+=h
            case=f'syn_{template}_{i:02d}';inst=beam.Instance(case,beam.ProblemVariant.PREMARSHALLING_CPMP,10,6,40,stacks)
            beam.create_initial_state(inst);code,n=beam._encode(stacks,10,6)
            if code in seen:raise ValueError('Independent generator duplicate canonical instance')
            seen.add(code);data=asdict(inst);data['variant']=inst.variant.value
            cases.append(dict(case=case,template=template,data_seed=seed,split='development' if i<4 else 'heldout',
                initial_n=n,instance=data,canonical_signature=code.hex()))
    return cases


def root_boundary_state(args):
    loc=solver_frame().f_locals;state={}
    for key,value in loc.items():
        if key in ('args','observer') or callable(value):continue
        state['run:'+key]=fingerprint(value)
    for key,value in vars(args).items():
        if key=='_union_selector_observer':continue
        if callable(value):raise TypeError('Callable in frozen solver args: '+key)
        state['args:'+key]=fingerprint(value)
    for key,value in vars(beam).items():
        if callable(value) and hasattr(value,'cache_info'):state['cache:'+key]=fingerprint(tuple(value.cache_info()))
    state['rng:python']=fingerprint(random.getstate());state['rng:numpy']=fingerprint(np.random.get_state())
    return state,loc['start']


class IndependentObserver(SelectorObserver):
    def __init__(self,clock,contract,expected=None):
        super().__init__('sa',capture_limit=0)
        self.clock=clock;self.contract=contract;self.expected=expected;self.cut=None;self.packet=None;self.originals={}
        self.context_layer=None;self.sample_layer=None;self.actual_frontier=None;self.frontier_verified=False

    def context(self,ranked,core,best_n):
        self.context_layer=solver_frame().f_locals['layer_index'];super().context(ranked,core,best_n)

    def sample(self,model,config,usable,target,linear,features,args,feature_sec,build_sec):
        self.sample_layer=args._layer_index;self.last_usable=list(usable);self.last_features=features;self.last_linear=list(linear)
        return super().sample(model,config,usable,target,linear,features,args,feature_sec,build_sec)

    def install(self):
        self.originals['_record_quality_reservoir_selected_telemetry']=beam._record_quality_reservoir_selected_telemetry
        beam._record_quality_reservoir_selected_telemetry=self.commit

    def eligible(self,selected_codes,args,loc):
        if self.context_layer!=loc['layer_index'] or self.sample_layer!=loc['layer_index']:return None
        if len(selected_codes)!=int(getattr(args,'beam_width',1200) or 1200):return None
        protected=set(self.layer_context[1]);sources=getattr(args,'_frontier_trace_last_selected_sources',{})
        positions=[i for i,c in enumerate(selected_codes) if c not in protected and sources.get(c)=='tail_fill'][-2:]
        if len(positions)!=2:return None
        macro_codes={c for (c,d),f in zip(self.last_usable,self.last_features) if beam._union_diversity_is_macro_feature(f)}
        pool=[c for c,d in self.last_usable if c in macro_codes and c not in selected_codes and d[0]>0 and d[0]+d[1]<=loc['cap']
            and not beam._parent_chain_would_cycle(loc['parent'],c,d[2])]
        if len(pool)<MIN_POOL:return None
        return positions,pool[:MAX_POOL],protected,macro_codes

    def build_packet(self,selected_codes,args,loc,choice):
        positions,pool,protected,macro_codes=choice;ranks={c:i for i,(c,d) in enumerate(self.layer_context[0])}
        byindex={c:i for i,(c,d) in enumerate(self.last_usable)};rows=[]
        for c in pool:
            d=loc['candidates'][c];p,moves,state=replay_prefix(loc['instance'],c,d,loc['parent'],loc.get('frg_macro_edge_paths',{}),
                getattr(args,'_frg_macro_edge_parent_by_code',{}))
            if not p['prefix_verified'] or len(moves)!=d[1]:raise ValueError('Independent chosen macro physical prefix failed')
            peak=beam.max_entropy_along_moves(loc['instance'],moves)
            if peak>loc['cap']:raise ValueError('Independent macro prefix exceeds cap')
            cached=getattr(args,'_union_rollout_probe_cache',{}).get(c);idx=byindex[c]
            edge=getattr(args,'_frg_macro_edge_moves_by_code',{}).get(c,())
            rows.append(dict(code=c.hex(),data=[d[0],d[1],d[2].hex()],union_index=idx,global_rank=ranks.get(c),
                native_linear=self.last_linear[idx],feature=encode(self.last_features[idx]._asdict()),prefix=p,prefix_peak_g_n=peak,
                macro_prefix_moves=[asdict(m) for m in edge],cached_rollout=asdict(cached) if cached is not None else None))
        surviving=[]
        for c in selected_codes:
            if c not in macro_codes:continue
            cached=getattr(args,'_union_rollout_probe_cache',{}).get(c);d=loc['candidates'][c]
            surviving.append(dict(code=c.hex(),data=[d[0],d[1],d[2].hex()],cached_rollout=asdict(cached) if cached is not None else None))
        return json.loads(json.dumps(dict(reference=[c.hex() for c in selected_codes],protected=[c.hex() for c in sorted(protected)],
            positions=positions,pool=rows,surviving_macros=surviving,macro_before=len(surviving),macro_after=len(surviving)+2,
            business_cap=loc['cap'],depth_cap=loc['depth_cap'],initial_n=args._initial_n,settings=scalar_settings(args),
            ranking_signature=fingerprint(self.layer_context[0]),candidates_signature=fingerprint(loc['candidates']),
            source_by_code={c.hex():getattr(args,'_frontier_trace_last_selected_sources',{}).get(c) for c in selected_codes}),ensure_ascii=False))

    def commit(self,selected_codes,args):
        if self.cut is None:
            loc=solver_frame().f_locals;choice=self.eligible(selected_codes,args,loc)
            if choice is not None:
                self.clock.mark_boundary();state,start=root_boundary_state(args);stats=fingerprint(self.contract.stats)
                if self.expected is not None:
                    mismatches=[k for k in sorted(set(state)|set(self.expected['state'])) if state.get(k)!=self.expected['state'].get(k)]
                    if mismatches:raise RuntimeError('Independent post-fill prefix drift: '+json.dumps(mismatches[:20]))
                    if stats!=self.expected['contract_stats_signature']:raise RuntimeError('Independent contract prefix drift')
                self.clock.continue_live();started=REAL_TIME();self.packet=self.build_packet(selected_codes,args,loc,choice)
                packet_sec=REAL_TIME()-started;packet_sig=fingerprint(self.packet)
                if self.expected is not None and packet_sig!=self.expected['packet_signature']:raise RuntimeError('Independent common macro packet drift')
                self.cut=dict(layer=loc['layer_index'],clock_calls=len(self.clock.values),cutoff=self.clock.cutoff,prefix_sec=self.clock.cutoff-start,
                    byteorder=sys.byteorder,state=state,state_signature=fingerprint(state),contract_stats_signature=stats,
                    packet_signature=packet_sig,packet_sec=packet_sec)
                self.expected_frontier=[(c,loc['candidates'][c][0],loc['candidates'][c][1]) for c in selected_codes]
                self.originals['_select_next_frontier']=beam._select_next_frontier;beam._select_next_frontier=self.check_next
                self.originals['CompactBeamResult']=beam.CompactBeamResult;beam.CompactBeamResult=self.result
                beam._record_quality_reservoir_selected_telemetry=self.originals['_record_quality_reservoir_selected_telemetry']
        return self.originals['_record_quality_reservoir_selected_telemetry'](selected_codes,args)

    def verify(self):
        actual=solver_frame().f_locals['frontier']
        if actual!=self.expected_frontier:raise RuntimeError('Independent native actual frontier differs')
        self.actual_frontier=[[c.hex(),n,d] for c,n,d in actual];self.frontier_verified=True

    def check_next(self,candidates,args):
        if not self.frontier_verified:self.verify()
        beam._select_next_frontier=self.originals['_select_next_frontier']
        return self.originals['_select_next_frontier'](candidates,args)

    def result(self,*args,**kwargs):
        if not self.frontier_verified:self.verify()
        return self.originals['CompactBeamResult'](*args,**kwargs)

    def restore(self):
        for name,value in self.originals.items():setattr(beam,name,value)


def freeze():
    if (OUT/'manifest.json').exists():raise FileExistsError('Independent panel already frozen')
    previous_manifest();cases=independent_instances()
    sources=['scripts/capture_independent_macro_windows.py','scripts/frg_rollout_coordinates.py','scripts/cpmp_budget_witness.py',
        'scripts/benchmark_compact_beam_qubo_n0.py','scripts/beam_selector_research.py','scripts/production_selector_replay.py',
        'scripts/probe_business_macro_windows.py','scripts/run_rollout_contract.py']
    write_json(OUT/'manifest.json',dict(date='2026-10-07',cases=cases,source_sha256={p:digest(ROOT/p) for p in sources},
        development=8,heldout=12,seed=37,request_sec=20.,alpha=1.3,min_pool=MIN_POOL,max_pool=MAX_POOL,watchdog_sec=120,
        environment=ENVIRONMENT,eligibility='First native post-fill1200 Beam with current observed union context, two unprotected tail slots and at least4 business-admissible cycle-safe macro outsiders. First4 in union order; no physical-check substitution.',
        no_activation='Record every no-eligible case, native final quality and complete time. No new seed/case replacement.',
        split='First4 of each independently shuffled height template development; remaining6 each heldout, fixed before outcomes.',
        identification_gate='Need >=3 independent development contexts and within-context variation before fitting a transferable pair predictor. Otherwise close identification negatively and report heldout coverage.',
        scope='New independent synthetic layouts, not canonical permutations/monotone relabelings of prior port cases; no claim of real-port heldout representativeness.'))


def manifest():
    m=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'));previous_manifest()
    for p,sha in m['source_sha256'].items():
        if digest(ROOT/p)!=sha:raise ValueError('Frozen independent source changed: '+p)
    return m


def worker(case):
    m=manifest();entry=next(x for x in m['cases'] if x['case']==case);inst=load_instance(entry['instance'])
    args=business_args();contract=CanonicalTailContract();clock=PrefixClock();observer=IndependentObserver(clock,contract)
    args._union_selector_observer=observer;random.seed(37);np.random.seed(37);started=REAL_TIME()
    try:contract.install();observer.install();time.perf_counter=clock;result=beam.run_compact_beam_instance(case,inst,args)
    finally:time.perf_counter=REAL_TIME;observer.restore();contract.restore()
    physical=REAL_TIME()-started;started=REAL_TIME()
    business_ok,goal_ok,msg=beam._validate_instance_result(result,inst,business_alpha=1.3,business_cap_enabled=True)
    audit=cache_audit(args);audit_sec=REAL_TIME()-started;path=[asdict(v) for v in result.moves]
    row=dict(case=case,split=entry['split'],template=entry['template'],capture_status='ELIGIBLE' if observer.cut is not None else 'NO_IDENTIFIABLE_WINDOW',
        valid=business_ok and goal_ok,moves=len(path) if goal_ok else None,status=result.status,validation_message=msg,
        physical_solver_sec=physical,validation_cache_audit_sec=audit_sec,measured_end_to_end_sec=physical+audit_sec,
        measured_end_to_end_within_budget=physical+audit_sec<=20.,cache_contract_invalid=audit['contract_invalid'],
        actual_frontier_verified=observer.frontier_verified if observer.cut is not None else None,
        boundary_layer=observer.cut['layer'] if observer.cut is not None else None,contract_stats=contract.stats,
        **{k:getattr(result,k) for k in TELEMETRY if hasattr(result,k)})
    write_json(OUT/'captures'/f'{case}.json',dict(row=row,path=path,calls=observer.records,actual_frontier=observer.actual_frontier,cache_audit=audit))
    if observer.cut is not None:
        if not observer.frontier_verified:raise RuntimeError('Independent captured frontier unverified')
        write_json(OUT/'prefixes'/f'{case}.json',dict(observer.cut,packet=observer.packet))
        p=OUT/'prefixes'/f'{case}.clock.gz';p.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(p,'wb') as s:s.write(clock.values.tobytes())
    if audit['contract_invalid']:raise RuntimeError('Independent canonical cache contract failed')
    print(json.dumps({k:row[k] for k in ['case','split','capture_status','valid','moves','physical_solver_sec','cache_contract_invalid']},ensure_ascii=False),flush=True)


def panel():
    m=manifest()
    for entry in m['cases']:
        case=entry['case'];p=OUT/'captures'/f'{case}.json'
        if p.exists():continue
        try:proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--case',case],cwd=ROOT,
            env={**os.environ,**ENVIRONMENT},capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=m['watchdog_sec'])
        except subprocess.TimeoutExpired:
            write_json(p,dict(row=dict(case=case,split=entry['split'],template=entry['template'],capture_status='EXTERNAL_WATCHDOG',valid=False,moves=None),path=[]));continue
        if proc.returncode:
            error=OUT/'errors'/f'{case}.json';write_json(error,dict(stdout=proc.stdout,stderr=proc.stderr));raise RuntimeError(proc.stderr)
        print(proc.stdout.strip(),flush=True)
    write_csv(OUT/'capture_results.csv',[json.loads(p.read_text(encoding='utf-8'))['row'] for p in sorted((OUT/'captures').glob('*.json'))])


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run','worker']);p.add_argument('--case');a=p.parse_args()
    if a.command=='freeze':freeze()
    elif a.command=='run':panel()
    else:worker(a.case)

if __name__=='__main__':main()
