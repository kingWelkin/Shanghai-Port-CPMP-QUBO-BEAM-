# v4.0 Flow Diagrams

> [!IMPORTANT]
> Historical v4.0 baseline. The promoted mainline is [v4.3](../../guides/v43-mainline-algorithm.md), validated at `1681/1681` with `105530` moves.

Date: 2026-07-03

These diagrams describe the current promoted profile:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish
```

## End-To-End Pipeline

```mermaid
flowchart TD
    A["Input CPMP instance"] --> B["Compact byte-state encoding"]
    B --> C["Layered beam search"]
    C --> D["Legal one-step successor generation"]
    D --> E["Layer metrics and feature signals"]
    E --> F["Candidate supply"]
    F --> G["Deterministic core selection"]
    F --> H["QUBO diversity supplement"]
    G --> I["Next beam"]
    H --> I
    I --> J{"N = 0 found?"}
    J -- "no" --> C
    J -- "yes" --> K["Backtrack selected compact lineage"]
    K --> L["Expand macro edges"]
    L --> M["Repair compact successor gaps"]
    M --> N["Safe post-polish"]
    N --> O["Lift compact path to crane moves"]
    O --> P["Validator replay"]
    P --> Q["Published solution row"]
```

## Candidate Supply Layer

```mermaid
flowchart LR
    A["Current beam parents"] --> B["Ordinary legal children"]
    A --> C["Wang feature candidates"]
    A --> D["FRG frontier children"]
    A --> E["Macro-chain witness prefixes"]
    A --> F["Unified macro supply prefixes"]

    B --> G["Shared candidate dictionary"]
    C --> G
    D --> G
    E --> G
    F --> G

    G --> H["Deduplicate by compact code"]
    H --> I["Record origin and macro path metadata"]
    I --> J["QUBO / deterministic selector"]
```

## QUBO Selector

```mermaid
flowchart TD
    A["Ranked legal/replayable candidates"] --> B["Deterministic core"]
    A --> C["Bounded QUBO pool"]
    C --> D["Linear quality energy"]
    C --> E["Pair diversity energy"]
    C --> F["Exact-k repair penalty"]
    D --> G["Simulated annealing QUBO sample"]
    E --> G
    F --> G
    G --> H["Cardinality repair"]
    B --> I["Selected next beam"]
    H --> I
```

Linear quality energy favors low `N`, low depth, Wang fixed-prefix/pair quality,
FRG transition quality, completion witness quality, receiver-debt control,
source-scarcity relief, and macro path evidence.

Pair diversity energy penalizes same-parent crowding, same-source crowding,
similar reduce signatures, same `N` band, repeated macro origin, and repeated
macro ancestry.

## Macro Candidate Lifecycle

```mermaid
sequenceDiagram
    participant Beam as Beam parent
    participant FRG as BS-FRG-style probe
    participant Pool as Candidate pool
    participant QUBO as QUBO selector
    participant Lift as Compact lift
    participant Val as Validator

    Beam->>FRG: bounded recovery-chain probe
    FRG-->>Pool: short prefix plus completion witness
    Pool->>QUBO: candidate with origin and macro path metadata
    QUBO-->>Pool: selected or discarded
    Pool->>Lift: selected compact macro edge
    Lift->>Lift: expand prefix and repair compact gaps
    Lift->>Val: physical crane moves
    Val-->>Beam: accept only if replay is valid
```

## Promotion Gate

```mermaid
flowchart TD
    A["New candidate profile"] --> B["Run open120 no-cap t120"]
    A --> C["Run paper BF/CV full1561 no-cap t120"]
    B --> D{"open120 = 120/120?"}
    C --> E{"BF = all solved?"}
    C --> F{"CV = all solved?"}
    D -- "no" --> X["Do not promote"]
    E -- "no" --> X
    F -- "no" --> X
    D -- "yes" --> G["Compare moves, penalty, runtime"]
    E -- "yes" --> G
    F -- "yes" --> G
    G --> H{"No new solved-count regression?"}
    H -- "no" --> X
    H -- "yes" --> I["Promote only with paired evidence"]
```

## What The Diagram Intentionally Excludes

There is no final "run BS-FRG fallback" branch and no "accept QUBO move" branch.
That is deliberate. v4.0 keeps a single solver path: compact beam search plus
QUBO survivor selection plus validator replay.
