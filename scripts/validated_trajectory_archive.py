"""Observe complete root trajectories without modifying native search choices."""
from __future__ import annotations
from dataclasses import replace
import inspect

import benchmark_compact_beam_qubo_n0 as beam
from container_reshuffle.solvers import bs_frg as frg
from frg_rollout_coordinates import move_core
from research_receiver_allocation import physical_replay
from production_selector_replay import REAL_TIME


class ValidatedTrajectoryArchive:
    def __init__(self,capacity=8,instance=None):
        self.capacity=capacity
        self.instance=instance
        self.paths={}
        self.originals={}
        self.stats=dict(observed_feasible=0,unique_seen=0,retained=0,ignored_subproblem=0,
                        replayed=0,observer_sec=0.)
        self.seen=set()

    def same_root(self,instance):
        return (self.instance is not None and instance.max_height==self.instance.max_height
                and tuple(map(tuple,instance.stacks))==tuple(map(tuple,self.instance.stacks)))

    def add(self,moves,origin):
        if self.instance is None:return
        started=REAL_TIME()
        normalized=tuple(replace(m,step=i) for i,m in enumerate(moves))
        self.stats['observed_feasible']+=1
        signature=tuple(move_core(m) for m in normalized)
        if signature not in self.seen:
            self.seen.add(signature)
            self.stats['unique_seen']+=1
            physical_replay(self.instance.stacks,self.instance.max_height,normalized,goal=True)
            self.stats['replayed']+=1
            self.paths[signature]=dict(moves=normalized,origin=origin)
            retained=sorted(self.paths,key=lambda s:(len(s),s))[:self.capacity]
            self.paths={s:self.paths[s] for s in retained}
        self.stats['retained']=len(self.paths)
        self.stats['observer_sec']+=REAL_TIME()-started

    def install(self):
        if self.originals:raise RuntimeError('Archive already installed')
        self.originals=dict(main=beam.run_compact_beam_instance,validator=beam._moves_reach_n0_with_cap,
                            cached=frg._frg_completion_cached)
        beam.run_compact_beam_instance=self.main
        beam._moves_reach_n0_with_cap=self.validate
        frg._frg_completion_cached=self.cached

    def restore(self):
        if not self.originals:return
        beam.run_compact_beam_instance=self.originals['main']
        beam._moves_reach_n0_with_cap=self.originals['validator']
        frg._frg_completion_cached=self.originals['cached']
        self.originals={}

    def main(self,case,instance,args):
        self.instance=instance
        return self.originals['main'](case,instance,args)

    def validate(self,instance,moves,cap):
        valid=self.originals['validator'](instance,moves,cap)
        if valid and self.same_root(instance):self.add(moves,'native_full_path_validation')
        return valid

    def cached(self,*args,**kwargs):
        result=self.originals['cached'](*args,**kwargs)
        if not result.feasible:return result
        caller=inspect.currentframe().f_back
        if caller.f_code is not frg.BsFrgSolver.solve.__code__:
            return result
        loc=caller.f_locals
        if not self.same_root(loc['instance']):
            self.stats['ignored_subproblem']+=1
            return result
        prefix=tuple(loc.get('child_moves',()))
        if len(prefix)!=kwargs.get('start_step',0):
            raise ValueError('Root recovery prefix does not match its actual start step')
        self.add(prefix+result.moves,'native_bs_frg_root_rollout')
        return result

    def entries(self):
        return [self.paths[s] for s in sorted(self.paths,key=lambda s:(len(s),s))]
