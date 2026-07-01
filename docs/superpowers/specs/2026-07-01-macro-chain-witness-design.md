# Macro Chain Witness Design

## Goal

Add an experimental profile that lets the current beam+QUBO union solver consume BS-FRG-style deep recovery chains as selectable macro candidates, without turning BS-FRG into a final fallback solver.

## Context

The current `scheduler_state_v5_qubo_diversity_union_macro_credit_v2_det_polish` profile is the main promotion candidate across `open120`, BF, and CV. It performs well overall, but large CV families such as `CV/10-10` remain unsolved because shallow FRG child injection rarely creates a completion witness. Trace evidence on `CV/10-10/data10-10-1` showed many FRG internal children, very few selected macro candidates, and zero terminal witnesses. A pure BS-FRG run can solve the same case, so the missing piece is not basic feasibility; it is a deep chain object with future proof.

## Non-Goals

- Do not replace the main solver with BS-FRG.
- Do not add a final BS-FRG fallback path that bypasses QUBO selection.
- Do not promote the new profile unless it improves the combined `open120`, BF, and CV scorecard.
- Do not add case-specific gates for individual instance names.

## Design

The new profile is `scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish`, routed by the benchmark variant `qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish`.

It builds on `scheduler_state_v5_qubo_diversity_union_macro_credit_v2_det_polish` and enables a new macro-chain witness supplier:

- Scan the best frontier parents under high-pressure layouts.
- Run a bounded BS-FRG completion rollout from each selected parent.
- If the rollout produces progress or a feasible completion, materialize a prefix of the rollout as a macro edge candidate.
- Store the macro edge path so normal parent backtracking can replay it.
- Store witness quality features so QUBO can reward candidates with future evidence and penalize long or cap-risky chains.

The QUBO layer remains responsible for survivor selection. Macro-chain candidates enter the same candidate dictionary as other beam children. They are not automatically grafted to the final answer and do not bypass `_select_next_frontier`.

## Candidate Object

A macro-chain witness candidate stores:

- final child layout code;
- parent frontier code;
- macro path codes from parent to child;
- chain length;
- rollout final `n`;
- rollout feasibility flag;
- witness quality in `[0, 1]`;
- cap risk based on `depth + chain_length + rollout_final_n`.

The first implementation reuses existing macro edge replay state where possible:

- `_frg_macro_edge_path_by_code`;
- `_frg_macro_edge_parent_by_code`;
- `candidate_origin_by_code`;
- candidate feature ledger fields already used by macro future credit.

## Activation

The first profile is intentionally conservative:

- active only for large/high-pressure layouts through structural thresholds, not case names;
- parent width defaults to a small value;
- per-parent completion time is bounded;
- materialized prefix length is bounded;
- QUBO selection remains deterministic.

## Testing

Focused unit tests must cover:

- unknown paper variants do not silently route to the default profile;
- the open runner routes the new variant to the new solver profile;
- a macro-chain witness candidate is added to the candidate pool with a macro edge path and parent binding;
- terminal witness-only behavior is not used by the new profile;
- macro-chain candidates can be selected by QUBO through the same normal frontier selection path.

Smoke verification must cover:

- focused tests in `tests/test_qubo.py`;
- `CV/10-10/data10-10-1` sample run against baseline, debt-basin, pure BS-FRG, and the new profile;
- a small `open120` smoke to check that the profile does not damage the current main benchmark class.

## Promotion Rule

The new profile is not promoted unless a paired run across `open120`, BF, and CV beats the current main profile. A single rescued large-CV case is not enough if it adds failures or large penalties elsewhere.
