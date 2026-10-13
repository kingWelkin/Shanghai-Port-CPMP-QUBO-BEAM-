"""Opt-in prefix binding repair; state-tail evidence remains independently reusable."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass

import benchmark_compact_beam_qubo_n0 as beam

PREFIX_FIELDS = ('_frg_macro_edge_path_by_code', '_frg_macro_edge_parent_by_code',
                 '_frg_macro_edge_moves_by_code', '_macro_alt_signature_by_code')
MISSING = object()


@dataclass(frozen=True)
class MacroEdge:
    parent: bytes
    child: bytes
    path: tuple[bytes, ...]
    moves: tuple


class MacroPrefixBinding:
    """Record generated edge versions and optionally protect/resolve prefix bindings.

    observe: record only, preserving the native search/prefix state exactly.
    guard: keep existing prefix metadata when a recorder only improves its tail.
    repair: guard plus recover an overwritten ancestor edge by its actual parent.
    Native source and defaults are not modified. Install after CanonicalTailContract.
    """
    def __init__(self, mode='observe'):
        if mode not in ('observe', 'guard', 'repair'):
            raise ValueError('Unknown prefix binding mode')
        self.mode = mode
        self.originals = {}
        self.edges = {}
        self.versions = {}
        self.stats = Counter()
        self.events = []
        self.shape = None

    def capture_one(self, args, code):
        shape = (getattr(args, '_stack_count', 0), getattr(args, '_max_height', 0))
        if all(shape):
            self.shape = shape
        paths = getattr(args, '_frg_macro_edge_path_by_code', {})
        parents = getattr(args, '_frg_macro_edge_parent_by_code', {})
        moves = getattr(args, '_frg_macro_edge_moves_by_code', {})
        path, parent = tuple(paths.get(code, ()) or ()), parents.get(code)
        if not path or parent is None or path[-1] != code:
            return
        record = MacroEdge(parent, code, path, tuple(moves.get(code, ()) or ()))
        key = (parent, code)
        old = self.edges.get(key)
        if old == record:
            return
        if old is not None:
            self.stats['same_edge_version_updates'] += 1
        self.edges[key] = record
        self.versions.setdefault(key, []).append(record)
        self.stats['edge_versions_recorded'] += 1

    def capture(self, args):
        for code in getattr(args, '_frg_macro_edge_path_by_code', {}):
            self.capture_one(args, code)

    def resolve_maps(self, raw_codes, paths, parents, moves=None):
        """Replace only missing/mismatched owner bindings using recorded edges.

        No search, move invention, or parent reassignment. Existing owner-matching
        records retain priority. Callers still validate full elementary paths.
        """
        resolved_paths, resolved_parents = dict(paths or {}), dict(parents or {})
        resolved_moves = dict(moves or {})
        restored = []
        for parent, child in zip(raw_codes, raw_codes[1:]):
            path = tuple(resolved_paths.get(child, ()) or ())
            if path and path[-1] == child and resolved_parents.get(child) == parent:
                continue
            record = self.edges.get((parent, child))
            if record is None:
                continue
            if self.shape is None or beam._compact_code_is_successor(parent, child, *self.shape):
                # An ordinary legal edge must not become an older, longer macro.
                continue
            resolved_paths[child] = record.path
            resolved_parents[child] = parent
            if record.moves:
                resolved_moves[child] = record.moves
            else:
                resolved_moves.pop(child, None)
            restored.append(dict(parent=parent.hex(), child=child.hex(), edge_cost=len(record.path)))
        return resolved_paths, resolved_parents, resolved_moves, restored

    def install(self):
        if self.originals:
            raise RuntimeError('Prefix binding already installed')
        original = beam._record_frg_macro_edge_ancestry
        self.originals['_record_frg_macro_edge_ancestry'] = original
        def ancestry(args, child_code, parent_code):
            result = original(args, child_code, parent_code)
            self.capture_one(args, child_code)
            return result
        beam._record_frg_macro_edge_ancestry = ancestry

        recorder = beam._record_macro_chain_witness_candidate
        self.originals['_record_macro_chain_witness_candidate'] = recorder
        def record(candidates, args, **kwargs):
            self.shape = (kwargs['stack_count'], kwargs['max_height'])
            layout_path = kwargs.get('layout_path') or ()
            code = None
            if layout_path:
                stacks = [list(s) for s in getattr(layout_path[-1], 'stacks', ())]
                if len(stacks) == kwargs['stack_count']:
                    code, _ = beam._encode(stacks, kwargs['stack_count'], kwargs['max_height'])
            old = candidates.get(code)
            saved = {name: getattr(args, name, {}).get(code, MISSING) for name in PREFIX_FIELDS}
            if code is not None:
                self.capture_one(args, code)
            result = recorder(candidates, args, **kwargs)
            if result is not None:
                self.capture_one(args, result)
            if old is not None and result == code and candidates.get(code) == old:
                changed = any(getattr(args, name, {}).get(code, MISSING) != saved[name] for name in PREFIX_FIELDS[:3])
                if changed:
                    self.stats['tail_only_prefix_overwrites'] += 1
                    if self.mode != 'observe':
                        for name, value in saved.items():
                            mapping = getattr(args, name, None)
                            if mapping is None:
                                continue
                            if value is MISSING:
                                mapping.pop(code, None)
                            else:
                                mapping[code] = value
                        self.capture_one(args, code)
                        self.stats['prefix_bindings_preserved'] += 1
                    if len(self.events) < 100:
                        self.events.append(dict(kind='tail_only', child=code.hex(),
                            retained_parent=old[2].hex(), incoming_parent=kwargs['parent_code'].hex(),
                            retained_depth=old[1], incoming_depth=kwargs['parent_depth'] + len(layout_path),
                            guarded=self.mode != 'observe'))
            return result
        beam._record_macro_chain_witness_candidate = record

        if self.mode == 'repair':
            backtrack = beam._backtrack_with_macro_edges
            self.originals['_backtrack_with_macro_edges'] = backtrack
            def recover(parent, target, macro_paths=None, *, macro_parent_by_code=None,
                        stack_count=None, max_height=None):
                if macro_paths is None:
                    return backtrack(parent, target, None, macro_parent_by_code=macro_parent_by_code,
                                     stack_count=stack_count, max_height=max_height)
                raw = beam._backtrack(parent, target)
                owners = macro_parent_by_code
                if owners is None:
                    # Preserve the native meaning of explicitly untagged input paths.
                    owners = {c: p for p, c in zip(raw, raw[1:]) if (macro_paths or {}).get(c)}
                paths, parents, _, restored = self.resolve_maps(raw, macro_paths, owners)
                self.stats['backtrack_calls'] += 1
                self.stats['backtrack_edges_restored'] += len(restored)
                return backtrack(parent, target, paths, macro_parent_by_code=parents,
                                 stack_count=stack_count, max_height=max_height)
            beam._backtrack_with_macro_edges = recover
            lift = beam._codes_to_moves_instance_with_macro_witnesses
            self.originals['_codes_to_moves_instance_with_macro_witnesses'] = lift
            def recover_moves(instance, raw_codes, macro_moves_by_code, macro_parent_by_code):
                parents, moves = dict(macro_parent_by_code), dict(macro_moves_by_code)
                restored = 0
                for parent, child in zip(raw_codes, raw_codes[1:]):
                    if moves.get(child) and parents.get(child) == parent:
                        continue
                    record = self.edges.get((parent, child))
                    if (record is not None and record.moves and not beam._compact_code_is_successor(
                            parent, child, instance.stack_count, instance.max_height)):
                        parents[child], moves[child] = parent, record.moves
                        restored += 1
                self.stats['macro_lift_edges_restored'] += restored
                return lift(instance, raw_codes, moves, parents)
            beam._codes_to_moves_instance_with_macro_witnesses = recover_moves

    def restore(self):
        for name, original in reversed(list(self.originals.items())):
            setattr(beam, name, original)
        self.originals.clear()
