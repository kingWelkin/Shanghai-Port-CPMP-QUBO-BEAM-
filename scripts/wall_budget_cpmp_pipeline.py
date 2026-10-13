"""Opt-in CPMP pipelines with native search and refinement inside one wall clock."""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict,replace
import random
import sys
import numpy as np

import research_paper_v43_panel as broad
import benchmark_compact_beam_qubo_n0 as beam
from container_reshuffle.solvers import bs_frg as frg
from container_reshuffle.solvers.base import SolverConfig
from container_reshuffle.core import Move
from beam_component_ablation import ComponentObserver,scalar_settings
from paper_priority_group_contract import PriorityGroupContract
from frg_rollout_coordinates import CanonicalTailContract
from macro_prefix_binding import MacroPrefixBinding
from validated_trajectory_archive import ValidatedTrajectoryArchive
from classical_trajectory_simplify import simplify
from trajectory_state_crossover import state_crossover
from research_receiver_allocation import physical_replay
from production_selector_replay import REAL_TIME,fingerprint
from hybrid_trajectory_refiner import no_milp
import bounded_conditioned_refinement as bounded
from capped_conditioned_refinement import run as capped

MODES=('native','bs_frg','frg','single','pair','qubo','oracle')


class NativeReserveReached(BaseException):
    """Leave native search only after a complete valid incumbent is available."""


class CompletePathLedger:
    def __init__(self,instance,start,budget,publish):
        self.instance=instance;self.start=start;self.deadline=start+budget;self.publish=publish
        self.best=None;self.seen=set();self.raw_best=None
        self.stats=dict(validated=0,rejected=0,graph_calls=0,graph_candidate_savings_sum=0,late=0)

    def commit(self,path,source):
        if self.best is not None and len(path)>=len(self.best):return False
        message=dict(type='INCUMBENT',source=source,moves=len(path),path=[asdict(m) for m in path],
                     ready_sec=REAL_TIME()-self.start)
        if REAL_TIME()>self.deadline:self.stats['late']+=1;return False
        self.best=tuple(path);self.publish(message);return True

    def offer(self,path,source):
        path=tuple(replace(m,step=i) for i,m in enumerate(path))
        signature=tuple((m.box,m.from_stack,m.to_stack) for m in path)
        if signature in self.seen:return False
        physical_replay(self.instance.stacks,self.instance.max_height,path,goal=True)
        self.seen.add(signature);self.stats['validated']+=1
        if REAL_TIME()>self.deadline:self.stats['late']+=1;return False
        accepted=self.commit(path,source)
        # Common output cleanup is charged equally to every method. It does
        # not replace an engine's internal search trajectory mid-pass.
        if self.raw_best is None or len(path)<self.raw_best:
            self.raw_best=len(path)
            cleaned,meta=state_crossover(self.instance.stacks,self.instance.max_height,[path])
            self.stats['graph_calls']+=1;self.stats['graph_candidate_savings_sum']+=len(path)-len(cleaned)
            accepted=self.commit(cleaned,'common_graph:'+source) or accepted
        return accepted


class NativeReserveGuard:
    def __init__(self,ledger,observer,cutoff,enabled):
        self.ledger=ledger;self.observer=observer;self.cutoff=cutoff;self.enabled=enabled
        self.originals=[];self.requests=0

    def request(self):
        self.requests+=1
        now=REAL_TIME() if self.requests%32==0 else None
        if (now is not None and self.enabled and now>=self.cutoff and self.ledger.best is not None
                and (self.observer is None or self.observer.pending_frontier is None)):
            raise NativeReserveReached

    def replace(self,module,name,value):
        self.originals.append((module,name,getattr(module,name)));setattr(module,name,value)

    def install(self):
        uncached=beam._successors_with_n_uncached;cached=beam._successors_with_n_cached
        def uncached_guard(*args,**kwargs):
            self.request();yield from uncached(*args,**kwargs)
        def cached_guard(*args,**kwargs):
            self.request();return cached(*args,**kwargs)
        cached_guard.cache_info=cached.cache_info;cached_guard.cache_clear=cached.cache_clear
        self.replace(beam,'_successors_with_n_uncached',uncached_guard)
        self.replace(beam,'_successors_with_n_cached',cached_guard)
        for name in ('_bg_choices','_reduction_choices','_single_move_children'):
            original=getattr(frg,name)
            def wrapped(*args,_original=original,**kwargs):
                self.request();return _original(*args,**kwargs)
            aliases=[key for key,value in vars(beam).items() if value is original]
            self.replace(frg,name,wrapped)
            for alias in aliases:self.replace(beam,alias,wrapped)

    def restore(self):
        for module,name,original in reversed(self.originals):setattr(module,name,original)
        self.originals=[]


class StreamingNativeArchive(ValidatedTrajectoryArchive):
    def __init__(self,instance,ledger):
        super().__init__(1,instance);self.ledger=ledger

    def add(self,moves,origin):
        # The existing collector binds root prefixes and validates coordinates.
        # Its de-duplication also avoids replaying repeated native notifications.
        before=self.stats['unique_seen'];super().add(moves,origin)
        if self.stats['unique_seen']!=before:self.ledger.offer(tuple(moves),'native:'+origin)


class StreamingBeamObserver(ComponentObserver):
    def __init__(self,start,publish):super().__init__();self.start=start;self.publish=publish

    def verify(self):
        pending=self.pending_frontier is not None
        super().verify()
        if pending:self.publish(dict(type='BEAM',elapsed_sec=REAL_TIME()-self.start,**self.commits[-1]))


def run(entry,mode,start,budget,publish,*,reserve=2.):
    if mode not in MODES:raise ValueError(mode)
    random.seed(37);np.random.seed(37)
    group=PriorityGroupContract();group.install()
    try:
        instance=broad.load_case(entry)
        ledger=CompletePathLedger(instance,start,budget,publish)
        archive=StreamingNativeArchive(instance,ledger)
        tails=CanonicalTailContract();binding=MacroPrefixBinding('guard')
        observer=StreamingBeamObserver(start,publish) if mode!='bs_frg' else None
        guard=NativeReserveGuard(ledger,observer,start+max(0.,budget-reserve),mode not in ('native','bs_frg'))
        native_status='RUNNING';native_result=None;switched=False
        args=broad.make_args(budget,37,instance,entry) if mode!='bs_frg' else None
        if args is not None:
            if getattr(args,'frg_seed_cache_input',None):raise ValueError('Historical incumbent cache is forbidden')
            args._union_selector_observer=observer
        publish(dict(type='PHASE',phase='native',elapsed_sec=REAL_TIME()-start,
                     reserve_sec=reserve if guard.enabled else 0.,mode=mode))
        try:
            tails.install();binding.install()
            if observer:observer.install()
            archive.install();guard.install()
            with redirect_stdout(sys.stderr):
                if mode=='bs_frg':
                    native_result=frg.BsFrgSolver(paper_pure=True).solve(instance,SolverConfig(
                        max_depth=1000,beam_width=1000,candidate_limit_per_state=3,
                        time_limit_sec=max(.001,start+budget-REAL_TIME()),entropy_alpha=10.,random_seed=37))
                else:native_result=beam.run_compact_beam_instance(entry['case'],instance,args)
            native_status=str(native_result.status)
            try:ledger.offer(tuple(native_result.moves),'native_final')
            except ValueError as error:
                ledger.stats['rejected']+=1
                publish(dict(type='REJECTED_NATIVE_FINAL',detail=str(error),elapsed_sec=REAL_TIME()-start))
        except NativeReserveReached:
            switched=True;native_status='REFINEMENT_RESERVE_SWITCH'
        finally:
            guard.restore();archive.restore()
            if observer:observer.restore()
            binding.restore();tails.restore()
        if observer and (observer.pending_frontier is not None or any(not c['actual_frontier_verified'] for c in observer.commits)):
            raise ValueError('Native phase left an unverified Beam commitment')
        native_end=REAL_TIME()-start
        native_path=ledger.best
        publish(dict(type='NATIVE_END',elapsed_sec=native_end,status=native_status,switched=switched,
            native_moves=None if native_path is None else len(native_path),
            native_path_signature=None if native_path is None else fingerprint([asdict(m) for m in native_path]),
            beam_checks=0 if observer is None else len(observer.commits),requests_observed=guard.requests,
            settings=None if args is None else scalar_settings(args)))
        if mode not in ('native','bs_frg') and ledger.best is not None and REAL_TIME()<ledger.deadline:
            publish(dict(type='PHASE',phase='common_simplification',elapsed_sec=REAL_TIME()-start))
            with no_milp():common,meta=simplify(instance.stacks,instance.max_height,ledger.best)
            ledger.offer(common,'common_simplification')
            publish(dict(type='COMMON_END',elapsed_sec=REAL_TIME()-start,moves=len(ledger.best),metadata=meta))
            if REAL_TIME()<ledger.deadline:
                def post_publish(message):
                    if message['type']=='INCUMBENT':
                        ledger.offer(tuple(Move(**v) for v in message['path']),'refine:'+message['source'])
                    else:publish(dict(message,stage='refinement'))
                mapped=dict(frg='frg',single='pair_frg',pair='pair_frg',qubo='capped_qubo_frg',oracle='capped_oracle_frg')[mode]
                old_seed=bounded.classical_seed
                try:
                    if mode=='single':bounded.classical_seed=lambda model,pairs=True:old_seed(model,pairs=False)
                    result=capped(instance,ledger.best,mapped,start,budget,post_publish,continue_frg=True)
                finally:bounded.classical_seed=old_seed
                ledger.offer(tuple(Move(**v) for v in result['path']),'refinement_final')
        return dict(mode=mode,valid=ledger.best is not None,moves=None if ledger.best is None else len(ledger.best),
            native_end_sec=native_end,elapsed_sec=REAL_TIME()-start,ledger=ledger.stats,global_cpmp_optimal=False)
    finally:group.restore()
