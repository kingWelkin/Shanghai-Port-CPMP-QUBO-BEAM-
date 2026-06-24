# Trace-First Open Residual Audit

Date: 2026-06-24

This note records the first trace-first audit after consolidating the algorithm
profile status. It focuses on the current mainline:

```text
qubo_scheduler_state_v5_qubo_diversity_union_polish
```

The purpose is diagnosis, not tuning. The audit asks whether the residual open
failures are caused by absent useful candidates, useful candidates not being
selected, or selected branches entering terminal dead regions.

## Fresh Run

Command:

```powershell
$env:PYTHONPATH='.'; python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test --case-ids 4,10 --variant qubo_scheduler_state_v5_qubo_diversity_union_polish --time-limit 10 --alpha 10 --jobs 1 --frontier-trace-top-k 128 --frontier-trace-output docs\experiments\trace_first_open_residual_frontier_20260624.csv --terminal-trace-output docs\experiments\trace_first_open_residual_terminal_20260624.csv --output docs\experiments\trace_first_open_residual_run_20260624.csv
```

Artifacts:

- `docs/experiments/trace_first_open_residual_run_20260624.csv`
- `docs/experiments/trace_first_open_residual_run_20260624_summary.md`
- `docs/experiments/trace_first_open_residual_frontier_20260624.csv`
- `docs/experiments/trace_first_open_residual_terminal_20260624.csv`

The CSV artifacts are ignored generated outputs. This guide preserves the
decision-level interpretation.

## Result Rows

| Case | Status | Final N | Moves | Runtime sec | QUBO selected on search path | QUBO ancestor on search path |
|---|---|---:|---:|---:|---:|---:|
| `5x5_cv1_test_4` | `TIME_LIMIT_OR_BEAM_EXHAUSTED` | 17 | none | 2.367 | 0 | 0 |
| `5x5_cv1_test_10` | `TARGET_REACHABLE` | 0 | 28 | 4.468 | 1 | 5 |
| `5x5_cv2_test_4` | `TARGET_REACHABLE` | 0 | 28 | 2.636 | 2 | 22 |
| `5x5_cv2_test_10` | `TIME_LIMIT_OR_BEAM_EXHAUSTED` | 15 | none | 1.770 | 0 | 0 |

The two failed rows keep the initial `final_n` because no full `N=0` path was
validated. The trace still shows the frontier reaching `N=1`.

## Layer-Level Findings

### Candidate Selection Is Not The Primary Blocker

Across both failed cases, improving candidates were generated and selected:

| Case | Trace layers | Layers with improving candidate | Layers where improving candidate was selected | Top candidate not selected |
|---|---:|---:|---:|---:|
| `5x5_cv1_test_4` | 33 | 16 | 16 | 0 |
| `5x5_cv2_test_10` | 29 | 14 | 14 | 0 |

The best available traced candidate was selected at every traced layer for both
failures. This rules out the simple hypothesis that QUBO diversity is discarding
the best ranked residual candidate.

### Both Failures Reach N=1

The failed cases follow a normal decreasing `N` trajectory:

- `5x5_cv1_test_4`: `17 -> ... -> 2` at layer 25, then `1` at layer 26.
- `5x5_cv2_test_10`: `15 -> ... -> 2` at layer 25, then `1` at layer 28.

The two solved controls also pass through `N=1`, then find a terminal `N=0`
child in the next expansion. That makes the residual failure more specific:
the mainline can find near-complete states, but the terminal transition is not
generated or preserved.

## Terminal Trace Findings

The terminal trace records every expanded `N=1` parent and counts whether its
legal children include `N=0`. This separates generation failure from pruning or
selection failure.

| Case | N=1 parent rows | Layers | N=0 children generated | N=0 children kept | Best generated child N |
|---|---:|---|---:|---:|---:|
| `5x5_cv1_test_4` | 4636 | 27-33 | 0 | 0 | 1 |
| `5x5_cv1_test_10` | 4 | 29 | 2 | 2 | 0 |
| `5x5_cv2_test_4` | 45 | 27 | 15 | 15 | 0 |
| `5x5_cv2_test_10` | 1 | 29 | 0 | 0 | 1 |

For the two failed cases, no `N=0` child is generated at all. Therefore the
terminal miss is not caused by QUBO selection, beam selection, business pruning,
or dominated-state pruning removing a reachable `N=0` child in these traced
runs.

Final-layer detail:

- `5x5_cv1_test_4` expands `1200` `N=1` parents at layer 33, generating
  `17616` children. The best generated child remains `N=1`; no child reaches
  `N=0`.
- `5x5_cv2_test_10` expands one `N=1` parent at layer 29, generating `12`
  children. The best generated child remains `N=1`; no child reaches `N=0`.

### The Two Residuals Fail Differently

`5x5_cv2_test_10` narrows sharply at the end:

- layer 26: `2142` candidates, `664` candidate parents;
- layer 27: `123` candidates, `45` candidate parents;
- layer 28: `3` candidates, `3` parents, all `N=1`;
- no `N=0` successor appears before beam exhaustion.

This looks like terminal basin collapse: the selected path reaches a small set
of `N=1` states whose legal children do not complete the layout.

`5x5_cv1_test_4` has a broader terminal plateau:

- layer 26 first reaches `N=1` with only `3` traced `N=1` candidates;
- layers 27 through 32 keep many traced `N=1` candidates;
- layer 32 still has `4285` candidates from `1162` candidate parents, all top
  traced candidates at `N=1`;
- no `N=0` successor appears.

This is not a narrow final collapse. It is a broad `N=1` plateau where the
available near-terminal states do not expose a legal final completion.

## Root-Cause Classification

Current classification:

| Case | Classification | Evidence |
|---|---|---|
| `5x5_cv1_test_4` | Selected near-terminal states enter a broad terminal plateau without `N=0` children. | Terminal trace covers 4636 expanded `N=1` parents across layers 27-33; none generates an `N=0` child. |
| `5x5_cv2_test_10` | Selected near-terminal states collapse into a tiny dead terminal basin. | Terminal trace covers the remaining layer-29 `N=1` parent; its 12 legal children all remain above `N=0`. |

This means the next fix should not start by increasing union QUBO weights or
adding more pair diversity. The first missing evidence is a terminal-transition
repair probe for `N=1` states whose one-step children cannot reach `N=0`.

## Recommended Next Step

Design a bounded terminal repair probe for selected `N=1` parents:

- trigger only when `current_n <= target_final_n + 1`;
- allow a short bridge of temporary moves before returning to `N=0`;
- cap expansions tightly and report success, bridge length, and moved stack
  sequence;
- graft the bridge only if the resulting full route validates under the same
  business cap.

The likely successor profile should be a near-terminal repair/escape operator,
not stronger global QUBO selection.

## Follow-Up: Terminal Bridge Probe

An opt-in profile was added to test the repair direction:

```text
qubo_scheduler_state_v5_qubo_diversity_union_terminal_bridge_polish
```

Command:

```powershell
$env:PYTHONPATH='.'; python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test --case-ids 4,10 --variant qubo_scheduler_state_v5_qubo_diversity_union_terminal_bridge_polish --time-limit 10 --alpha 10 --jobs 1 --output docs\experiments\terminal_bridge_open_residual_run_20260624.csv
```

Result:

| Case | Final N | Moves | Valid | Runtime sec | Terminal bridge calls | Terminal bridge successes | Terminal bridge moves |
|---|---:|---:|---|---:|---:|---:|---:|
| `5x5_cv1_test_4` | 0 | 31 | yes | 4.012 | 100 | 1 | 3 |
| `5x5_cv1_test_10` | 0 | 28 | yes | 4.196 | 0 | 0 | 0 |
| `5x5_cv2_test_4` | 0 | 28 | yes | 2.349 | 0 | 0 | 0 |
| `5x5_cv2_test_10` | 15 | none | no | 1.932 | 2 | 0 | 0 |

The bridge probe converts the broad terminal plateau case
`5x5_cv1_test_4` from unsolved to valid `N=0`, with a 3-move terminal bridge.
The solved controls remain valid and do not need the bridge.

A deeper single-case probe for `5x5_cv2_test_10` also failed:

```powershell
$env:PYTHONPATH='.'; python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv2_test --case-ids 10 --variant qubo_scheduler_state_v5_qubo_diversity_union_terminal_bridge_polish --time-limit 10 --alpha 10 --jobs 1 --terminal-bridge-depth 6 --terminal-bridge-max-expanded 20000 --terminal-bridge-max-seed-candidates 256 --output docs\experiments\terminal_bridge_cv2_10_depth6_20260624.csv
```

It still produced `terminal_bridge_calls=2`, `terminal_bridge_successes=0`,
and `terminal_bridge_expanded_total=2`. That residual is not simply a
too-shallow terminal bridge. It needs an earlier basin escape or reserve policy
before the frontier collapses to the final `N=1` parent set.
