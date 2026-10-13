"""Capture naturally nontrivial macro-selector calls from the existing general route."""
from __future__ import annotations
import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'scripts',ROOT/'src'):
    sys.path.insert(0,str(path))
import benchmark_compact_beam_qubo_n0 as beam
import benchmark_compact_paper_set as paper
import open_cpmp_benchmarks as opened
from beam_selector_research import SelectorObserver,encode
from run_beam_selector_research import ENVIRONMENT,digest,make_args,write_csv,write_json

OUT=ROOT/'docs/experiments/beam_selector_round5_20260924'


class MacroWindowObserver(SelectorObserver):
    def __init__(self):
        super().__init__('sa',capture_limit=0)
        self.windows=[]

    def sample(self,model,config,usable,target,linear,features,args,feature_sec,build_sec):
        sample_set=super().sample(model,config,usable,target,linear,features,args,feature_sec,build_sec)
        macro_count=sum(bool(f.is_frg_macro) for f in features)
        self.current.update(macro_count=macro_count,layer=int(getattr(args,'_layer_index',-1)),
            core_count=len(self.layer_context[1]) if self.layer_context else 0,
            eligible=len(usable)>target and macro_count>0)
        self.macro_pending=None
        if self.current['eligible'] and len(self.windows)<2:
            details=[]
            for (code,data),feature in zip(usable,features):
                moves=tuple((getattr(args,'_frg_macro_edge_moves_by_code',{}) or {}).get(code,()))
                path=tuple((getattr(args,'_frg_macro_edge_path_by_code',{}) or {}).get(code,()))
                parent=(getattr(args,'_frg_macro_edge_parent_by_code',{}) or {}).get(code)
                details.append(dict(code=code,data=data,feature=feature,
                    macro_parent=parent,macro_parent_matches=parent==data[2],
                    prefix_moves=[asdict(m) for m in moves],prefix_codes=path))
            self.macro_pending=dict(schema='production-macro-window/v1',call=self.current['call'],
                layer=self.current['layer'],target=target,linear=linear,model=model.to_dict(),
                core_codes=list(self.layer_context[1]),candidates=details,
                stack_count=args._stack_count,max_height=args._max_height,
                contract='Static production window; not a full continuation checkpoint')
        return sample_set

    def finish(self,ordered,selector_sec):
        super().finish(ordered,selector_sec)
        if self.macro_pending is not None:
            self.macro_pending['stages']=dict(self.current)
            self.windows.append(self.macro_pending)


def freeze():
    if (OUT/'manifest.json').exists():
        raise FileExistsError('Already frozen')
    previous=json.loads((ROOT/'docs/experiments/beam_selector_20260922/baseline_manifest.json').read_text(encoding='utf-8'))
    cases=list(previous['cases'])
    extras=paper.discover_sampled_paper_cases(ROOT/'benchmarks/paper',datasets=['BF'],families=['BF11','BF16'],sample_per_family=1)
    for c in extras:
        cases.append(dict(case=c.case_id,dataset=c.dataset,family=c.family,
            path=c.instance_path.relative_to(ROOT).as_posix(),max_height=None,sha256=digest(c.instance_path)))
    assert len(cases)==12
    sources=['scripts/probe_production_macro_windows.py','scripts/benchmark_compact_beam_qubo_n0.py',
             'scripts/beam_selector_research.py','scripts/run_beam_selector_research.py']
    write_json(OUT/'manifest.json',dict(cases=cases,seed=37,cap_sec=20.0,process_watchdog_sec=50,
        profile=make_args(20,37).solver_profile,beam_width=1200,core_width=1080,
        source_sha256={p:digest(ROOT/p) for p in sources},environment=ENVIRONMENT,
        capture_rule='first two natural union calls with n>k and macro_count>0 per case',
        scope='Existing v4.3 general route only; router not exercised; no width/core/generator tuning'))


def manifest():
    m=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    for path,sha in m['source_sha256'].items():
        if digest(ROOT/path)!=sha: raise ValueError(f'Source changed: {path}')
    for c in m['cases']:
        if digest(ROOT/c['path'])!=c['sha256']: raise ValueError('Input changed')
    return m


def worker(case_id):
    m=manifest();c=next(c for c in m['cases'] if c['case']==case_id)
    instance=paper.load_bf_bay(ROOT/c['path']) if c['dataset']=='BF' else opened.load_dlts_dat(ROOT/c['path'],max_height=c['max_height'])
    args=make_args(m['cap_sec'],m['seed'])
    observer=MacroWindowObserver();args._union_selector_observer=observer
    start=time.perf_counter()
    result=beam.run_compact_beam_instance(case_id,instance,args)
    wall=time.perf_counter()-start
    _,valid,msg=opened._open_validate_instance_result(result,instance,10.0,business_cap_enabled=False)
    row=dict(case=case_id,family=c['family'],valid=valid,moves=len(result.moves) if valid else None,
        solver_wall_sec=wall,within_budget=wall<=m['cap_sec'],union_calls=len(observer.records),
        nontrivial_calls=sum(r['n']>r['k'] for r in observer.records),
        macro_calls=sum(r['macro_count']>0 for r in observer.records),
        eligible_calls=sum(r['eligible'] for r in observer.records),saved_windows=len(observer.windows),
        expanded=result.expanded,generated=result.generated,status=result.status,validation_message=msg)
    write_json(OUT/'runs'/f'{case_id}.json',dict(row=row,path=[asdict(move) for move in result.moves],calls=observer.records))
    for i,window in enumerate(observer.windows):
        window.update(case=case_id,family=c['family'])
        path=OUT/'windows'/f'{case_id}__{i}.json.gz'
        path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(path,'wt',encoding='utf-8') as stream:
            json.dump(encode(window),stream,separators=(',',':'))
    print(json.dumps({k:v for k,v in row.items() if k not in ('status','validation_message')},ensure_ascii=False),flush=True)


def panel():
    m=manifest()
    for c in m['cases']:
        target=OUT/'runs'/f"{c['case']}.json"
        if target.exists(): continue
        try:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--case',c['case']],
                cwd=ROOT,env={**os.environ,**ENVIRONMENT},timeout=m['process_watchdog_sec'],capture_output=True,text=True,encoding='utf-8',errors='replace')
            if result.returncode:
                write_json(OUT/'errors'/f"{c['case']}.json",dict(stderr=result.stderr,stdout=result.stdout,returncode=result.returncode))
                raise RuntimeError(result.stderr)
            print(result.stdout.strip(),flush=True)
        except subprocess.TimeoutExpired:
            write_json(target,dict(row=dict(case=c['case'],family=c['family'],valid=False,moves=None,
                status='PROCESS_WATCHDOG',solver_wall_sec=None,within_budget=False,eligible_calls=None),calls=[],path=[]))
            print(c['case']+' PROCESS_WATCHDOG',flush=True)
    rows=[json.loads(p.read_text(encoding='utf-8'))['row'] for p in sorted((OUT/'runs').glob('*.json'))]
    write_csv(OUT/'activation_results.csv',rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run','worker']);p.add_argument('--case')
    args=p.parse_args()
    if args.command=='freeze': freeze()
    elif args.command=='run': panel()
    else: worker(args.case)

if __name__=='__main__': main()
