# Current Mainline Algorithm

Date: 2026-07-10
Version: v4.3

The current mainline is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_wang_route_v43_polish
```

v4.3 is a deterministic pre-solve route over two proven profiles. It sends
small DLTS-style layouts to Wang pair-QUBO with successor-cache polishing and
sends the remaining surface to the v4.2 chain-aware QUBO beam with bounded
BS-FRG short-incumbent compression. It never waits for one solver to fail before
starting another.

The full promotion gate is `1681/1681` with `105530` moves: Open120 `120/120`,
BF `721/721`, and CV `840/840`. Real30 remains `30/30` with `525` moves.

See [v4.3 mainline algorithm](v43-mainline-algorithm.md) for mechanism,
pseudocode, evidence, and reproduction commands.
