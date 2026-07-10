# Feasibility And DLTS Continuation Design

Date: 2026-06-15

## Purpose

Continue the interrupted paper-engineering work on top of the current best CPMP line without replacing it. The current quality mainline is `qubo_scheduler_state_v5_wang_rescue_polish`; this design adds a focused feasibility escape layer inspired by Wang, Jin, Zhang, and Lim (2017), and adds a DLTS-ready policy/value interface inspired by Hottung, Tanaka, and Tierney (2019).

The first implementation should improve or explain the two remaining hard open cases:

- `5x5_cv1_test_4`
- `5x5_cv2_test_10`

The main metric remains full-case quality: valid count, invalid count, full-quality penalty, solved moves, reference gap, runtime, and QUBO counters.

## Paper Mapping

### Feasibility-Based Heuristic

Paper: Wang, Jin, Zhang, and Lim, "A feasibility-based heuristic for the container pre-marshalling problem", EJOR 256 (2017), pp. 90-101.

Useful concepts for this codebase:

- State feasibility: reject or penalize candidate moves that consume slack needed to finish fixing containers.
- Container stability: prefer moves that keep or increase stable sorted prefixes.
- Dead-end avoidance: detect search states that have low slack and no plausible escape move.
- Tier-protection: avoid disturbing already useful low-tier structure.
- Task-type discrimination: treat direct fix, blocker clearing, temporary slack creation, and cleanup differently.

This is not a full reproduction of the paper's constructive heuristic. It is an explicit feature and escape layer inside the existing compact parent-chain solver.

### DLTS

Paper: Hottung, Tanaka, and Tierney, "Deep Learning Assisted Heuristic Tree Search for the Container Pre-marshalling Problem", COR 2019.

Useful concepts for this codebase:

- Policy signal: rank child states or moves.
- Value signal: estimate remaining work for pruning or rescue triggers.
- Offline supervised examples from near-optimal or best-known paths.
- Search integration by branching and bounding, not by replacing the solver.

The first implementation will not train or load a neural model. It will provide a deterministic policy/value adapter and a feature-export path so a real model can be introduced later without changing compact search contracts.

## Architecture

The current compact byte-state parent chain remains the source of truth. New mechanisms must only add candidate scores, gates, or legal compact child states.

The design adds two narrow units:

1. `container_reshuffle/solvers/feasibility_signals.py`
   - Computes feasibility and DLTS-ready feature signals from stack layouts.
   - Has no dependency on benchmark runners or QUBO samplers.

2. `scripts/benchmark_compact_beam_qubo_n0.py`
   - Consumes those signals through explicit experimental profiles.
   - Adds a bounded feasibility escape selector for residual hard cases.
   - Emits telemetry and optional supervised-example rows.

This keeps research logic testable outside the large benchmark script while preserving the existing runner surface.

## Feasibility Signals

Add a compact, deterministic signal object:

```python
@dataclass(frozen=True)
class FeasibilitySignals:
    fixed_prefix_total: int
    stable_prefix_total: int
    target_gap: int
    empty_slots: int
    blocked_fixed_candidates: int
    temporary_slack_moves: int
    low_tier_risk: int
    dead_end_risk: float
```

The initial implementation should use existing layout semantics:

- A stable prefix is the bottom-up sorted prefix in each stack.
- `target_gap` reuses the existing target-layout gap idea where available.
- `empty_slots` is total residual bay capacity.
- `blocked_fixed_candidates` counts top-blocked boxes that could contribute to a stable prefix if cleared.
- `temporary_slack_moves` counts legal top moves into stacks that do not increase `N`.
- `low_tier_risk` counts moves that would disturb low-tier sorted structure.
- `dead_end_risk` is a simple normalized score from low slack, high unresolved `N`, and low temporary-slack availability.

All formulas must be conservative and deterministic. The first goal is useful ranking and telemetry, not a mathematically complete feasibility theorem.

## Feasibility Escape

Add one experimental profile:

```text
scheduler_state_v5_feasibility_escape_polish
```

Add one open-runner alias:

```text
qubo_scheduler_state_v5_feasibility_escape_polish
```

The profile extends `scheduler_state_v5_wang_rescue_polish` and enables a bounded escape layer only when:

- best `N` has plateaued,
- current layer is near the depth cap,
- parent concentration or dead-parent ratio is high,
- `dead_end_risk` is above threshold,
- a temporary-slack candidate exists.

The escape selector may add only legal compact successor states from the current frontier. It must not splice externally generated move sequences. If a Wang BS-B reference is used, it is used only as diagnostic teacher data, not as a direct path source.

## DLTS Adapter

Add a model-free adapter first:

```python
class DltsSignalAdapter:
    def policy_score(self, features: DltsStateFeatures) -> float: ...
    def value_estimate(self, features: DltsStateFeatures) -> float: ...
```

The default implementation is deterministic:

- policy rewards lower `N`, lower depth, higher stable prefix, more temporary slack, and lower dead-end risk.
- value estimates remaining work as a weighted lower-bound-like score.

The adapter exists so the later neural model can be loaded behind the same interface. The first implementation should not add TensorFlow, PyTorch, Keras, or model-file requirements.

## Data Export

Add an optional CSV export flag to the compact runner:

```text
--dlts-feature-output docs/experiments/scratch_dlts_features.csv
```

Rows should describe candidate states from selected layers:

- case id
- depth
- parent rank
- `N`
- feasibility signals
- whether the candidate was selected
- whether it lies on the final reconstructed path, when known
- final case status

This export is for future supervised learning and audit, not for current promotion.

## Testing

Use TDD for all implementation:

- Unit tests for `FeasibilitySignals` on tiny layouts.
- Unit tests that the escape score prefers temporary-slack creation over low-tier damage.
- Profile tests proving defaults remain unchanged and the new profile explicitly enables feasibility escape.
- Open-runner alias tests for the new variant.
- Export tests proving CSV headers and rows are stable.
- Focused smoke on the two residual failures before broader runs.

Recommended first verification:

```powershell
python -m pytest tests/test_feasibility_signals.py tests/test_qubo.py::test_open_runner_accepts_feasibility_escape_variant -q --basetemp .pytest_tmp_feas_escape
```

## Benchmark Plan

Start narrow:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test --case-ids 4,10 --variant qubo_scheduler_state_v5_feasibility_escape_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_feasibility_escape_residual.csv
```

Then compare a clean 120-case run only if the residual smoke improves or gives clearer diagnostics:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_feasibility_escape_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_feasibility_escape_open120.csv
```

Promotion requires beating `qubo_scheduler_state_v5_wang_rescue_polish` on full-case quality or solving one residual case without adding new failures in a clean paired run.

## Non-Goals

- Do not make the new profile default.
- Do not train a DLTS neural model in the first pass.
- Do not add heavyweight ML dependencies.
- Do not call original DLTS from inside compact search.
- Do not use Wang BS-B or DLTS paths as direct move splices.
- Do not promote results based only on solved-case move totals.

## Acceptance Criteria

- Existing default profiles keep behavior.
- New behavior is behind explicit profile and CLI flags.
- Compact parent-chain reconstruction remains the only path source.
- Targeted unit tests and profile tests pass.
- Focused residual-case smoke produces either an improved solve or a documented negative result.
- Curated experiment notes state clearly whether feasibility escape is promoted, kept experimental, or abandoned.
