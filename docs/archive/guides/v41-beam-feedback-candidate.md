# v4.1 Beam Feedback Candidate

> [!IMPORTANT]
> Historical document. It does not define the current default. The promoted mainline is [v4.3](../../guides/v43-mainline-algorithm.md), validated at `1681/1681` with `105530` moves.

Date: 2026-07-07

Profile:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_feedback_v41_polish
```

Baseline:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish
```

## Position

This is an experimental v4.1 mainline candidate, not a promoted replacement for
v4.0. It keeps the single QUBO-beam solver path:

- no BS-FRG post-compress;
- no solver portfolio;
- no learning model;
- no external QUBO API.

The goal is to improve feedback inside the beam selector before any feasible
solution exists.

## Mechanism

The candidate adds a conservative beam-feedback credit layer:

- compare witness alternatives only within the same parent group;
- compute credit from witness quality, progress, source scarcity, survival
  credit, receiver debt, relative witness cost, and height risk;
- apply only high-confidence positive credit to QUBO linear energy;
- avoid turning negative feedback into a global QUBO penalty;
- keep pair-signature redundancy penalty available as an ablation knob, but
  disabled in the retained profile;
- scope feedback preparation to the QUBO usable candidate window;
- keep the global quality reservoir disabled in this profile.

This shape is deliberately narrower than the first implementation attempt. The
broad reservoir version solved `0/5` on `CV/10-6` first five cases at `t120`,
so the retained profile favors stability over aggressive candidate reshaping.

## Focused Smoke Evidence

These focused runs explain why the profile was worth full-gate testing, but
they are not promotion evidence by themselves.

| surface | v4.0 solved | v4.0 moves | v4.0 avg runtime sec | v4.1 solved | v4.1 moves | v4.1 avg runtime sec | move delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| open `5x5_cv1_test` + `5x5_cv2_test`, first 5 each, `t10` | 10/10 | 707 | 0.490 | 10/10 | 707 | 0.503 | 0 |
| paper CV `10-6`, first 5, `t120` | 5/5 | 2024 | 3.511 | 5/5 | 1985 | 3.737 | -39 |
| paper BF hard sample `BF14,BF16,BF30,BF32`, first 1 each, `t120` | 4/4 | 520 | 7.569 | 4/4 | 507 | 7.697 | -13 |

## Full Gate Check

The first full gate check did not pass.

| surface | v4.0 reference solved | v4.0 reference moves | v4.1 solved | v4.1 moves | v4.1 full penalty | decision |
|---|---:|---:|---:|---:|---:|---|
| open120 DLTS, no-cap `t120` | 120/120 | 6928 | 120/120 | 7065 | 7065 | solved, but longer |
| paper BF, no-cap `t120`, `--sample-per-family 9999` | 721/721 | 46517 | 720/721 | 47435 | 47641 | failed solved-count gate |
| paper CV, no-cap `t120`, `--sample-per-family 9999` | 840/840 | 60386 | 838/840 | 59707 | 60565 | failed solved-count gate |
| paper BF/CV combined | 1561/1561 | 106903 | 1558/1561 | 107142 | 108206 | failed solved-count gate |

Failed full-gate rows:

- `BF/BF5/cpmp_16_5_64_13_39_13`, final_n `39`, runtime `121.689s`.
- `CV/10-10/data10-10-15`, final_n `80`, runtime `120.838s`.
- `CV/10-10/data10-10-28`, final_n `82`, runtime `120.585s`.

Interpretation: the conservative feedback layer is stable enough to complete
most cases, and it can shorten selected hard samples, but it still perturbs
some recovery chains that v4.0 solves. That makes it a useful research branch,
not a v4.1 promotion candidate yet.

## Promotion Gate

Do not promote this profile unless a paired full run against v4.0 satisfies all
of the following:

- open120 DLTS no-cap `t120`: `120/120`;
- paper BF no-cap `t120`, `--sample-per-family 9999`: `721/721`;
- paper CV no-cap `t120`, `--sample-per-family 9999`: `840/840`;
- combined moves lower than the same-day v4.0 paired baseline;
- open120, BF, and CV are reported separately.
