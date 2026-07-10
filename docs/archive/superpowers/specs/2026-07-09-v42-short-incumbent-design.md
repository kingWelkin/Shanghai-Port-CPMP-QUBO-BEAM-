# v4.2 Short Incumbent Design

## Goal

Add an experimental v4.2 profile that keeps the v4.0 chain-aware solver as the coverage anchor, then uses BS-FRG as a validated shorter incumbent source to reduce path length without reducing solved count.

## Success Criteria

- Open120 remains 120/120 solved.
- Paper BF remains 721/721 solved.
- Paper CV remains 840/840 solved.
- Combined moves must be materially below v4.0 stable's 114331, with the first target below 100000.
- A shorter BS-FRG path is accepted only after replay validation reaches final_n=0.

## Design

The first v4.2 candidate reuses the existing `bs_frg_postcompress` mechanism as a long-path compressor. Focused probes showed that small Open120 cases are already near reference length, so the profile avoids adding a fixed postcompress tax there. It triggers on paths with at least 120 moves, uses a 4 second BS-FRG budget, and passes the current v4 solution into BS-FRG as an incumbent bound so the BS-FRG search can prune any path that cannot beat the current validated path.

This is not a portfolio default. The v4.0 path remains the safety anchor. BS-FRG is only a shorter incumbent producer, and the final result changes only when the candidate is legal and strictly shorter.

## Implementation Boundary

- Add a new profile named `scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_short_incumbent_v42_polish`.
- Add runner variant `qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_short_incumbent_v42_polish`.
- Preserve existing v4.0 and quality profiles unchanged.
- Use existing telemetry plus one explicit postcompress incumbent-bound switch for traceability.
