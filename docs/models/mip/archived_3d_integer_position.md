# Archived 3D Integer-Position MIP

`mip_3d` / `Solve3DIntegerPositionMip` is retained for reproducibility and model-comparison experiments, but it is no longer the recommended exact benchmark for the current business objective.

## Why It Is Archived

The current CPMP business objective depends on `N(Future)`, which is defined from adjacent levels inside each stack. A pure 3D formulation with container-stack-time assignment `a[p,i,t]` and integer level `lev[p,t]` does not directly know which container occupies slot `(i,h,t)`.

To model `N(Future)` exactly, the implementation has to introduce auxiliary slot variables. Once those variables are present, the 3D model carries both:

- the weaker big-M level/position machinery of the 3D model, and
- slot-like variables needed for adjacent-level disorder calculation.

On the current medium benchmark, the enhanced 3D model still proves optimality but remains slower than 4D:

| Model | Vars | Constraints | Nodes | Runtime | Result |
|---|---:|---:|---:|---:|---|
| 3D enhanced | 4236 | 7976 | 510 | 24.2s | optimal 8 |
| 4D enhanced | 7388 | 13683 | 37 | 8.8s | optimal 8 |

## Current Policy

- Keep `mip_3d` callable for historical experiments and regression tests.
- Do not include `mip_3d` in new default experiment configs.
- Use `mip_4d` as the primary exact benchmark for small CPMP cases under the `G=M+N(Future)` business constraint.
- Treat 3D results as secondary diagnostics, not as the main solver line.
