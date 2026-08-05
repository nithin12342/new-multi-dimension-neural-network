# Core Principles (the "What" — never violated)

These hold regardless of project, language, or team size. Everything in `execution-algorithm.md` and `state-machine.md` is a mechanical consequence of these, not an independent rule set.

## 1. Structure before detail (Skeleton-of-Thought)
Never generate implementation detail before the structure that contains it exists and has been checked. Skeleton -> folders -> file responsibilities -> code skeleton (signatures only) -> logic. Skipping a layer to "save time" is the single most common cause of architectural drift.

## 2. Domain shapes the modules (DDD)
Bounded contexts, aggregates, and invariants are decided before any file exists. A bounded context becomes a top-level module/crate. An aggregate becomes a file or small file-group. Nothing is folded together for convenience.

## 3. SOLID is enforced by file structure, not by convention
| Principle | File-structure consequence |
|---|---|
| **S**RP | One file = one reason to change. If you can't name the file's single responsibility in <=7 words, split it. |
| **O**CP | New behavior = new files implementing a shared trait/interface. `core/` is closed, `impls/`/`plugins/` is open. |
| **L**SP | Files implementing the same trait/interface live in the same directory, substitutable by running the same fixture against each. |
| **I**SP | Traits/interfaces split into the smallest coherent files. |
| **D**IP | `domain/` depends on nothing. `application/` depends only on `domain/` trait defs. `infrastructure/` implements those traits. Import arrows only point inward — enforced by module privacy (a compile error), not a lint. |

## 4. Verification-first: nothing is done on a claim
"It should work" is not a valid state. A node is done only when (a) it compiles clean or with justified warnings, and (b) a real execution against real input produced the expected artifact. No unit tests — see Execution Algorithm for why E2E-only is the testing model here.

## 5. No phase advances without satisfying its exit gate
See `state-machine.md`. A failed gate returns to the phase that produced the failure, it never "continues anyway."

## 6. The Skeleton is a living artifact, never a stale plan
Any node whose implementation reveals a new invariant, a missing port, or a wrong boundary updates the Skeleton document in the same commit that fixes the code. Planning documents that drift from the code they describe are worse than no documentation, because they're actively misleading.

## 7. Minimum-correction principle
When verification fails, fix the smallest thing that makes it pass — a diff/patch, not a file rewrite. A full-file rewrite destroys the verification audit trail and usually reintroduces drift from the Skeleton. Only rewrite a file wholesale if the Skeleton itself changed that file's responsibility.

## 8. Idempotency
Every script that generates files, runs migrations, or re-executes a node's build/test must be safe to re-run: check-before-write, no append-only side effects that duplicate on re-run. Side-effecting node implementations get an explicit "run twice, confirm identical resulting state" check as part of their verification evidence.

## 9. ACID applied to the build process itself
- **Atomicity** — a node's change lands whole or not at all; a failed gate reverts to the last accepted commit.
- **Consistency** — the Skeleton and the codebase must never disagree; checked at every gate.
- **Isolation** — one node worked per branch/worktree; in-progress stub-breaking changes don't block parallel planning.
- **Durability** — every accepted node is committed with its verification evidence attached (compiler output + runtime output), forming the audit trail.

## 10. Traceability is mandatory, not optional documentation
Every artifact — requirement, spec, skeleton node, folder, file, class, method, verification record — carries an ID, and the chain between them is never broken. See `node-schema.md`. This is what makes change-impact analysis and audits possible instead of aspirational.

## 11. Goal-scoped, chain-traced, complete correction — no collateral damage, no missing chain
Before touching any code to fix an error, state the goal of the correction explicitly. Trace the chain of implicated files using their stated responsibilities. Change only that location. Minimal means the smallest diff that resolves the entire causal chain.

## 12. No Mock Fallouts or Mock Fallbacks — Authentic Data Only
There is only one way: authentic, real data only. No synthetic/mock data generation, no dummy tensor fallbacks, and no mock fallouts. All datasets must be downloaded, preprocessed, and loaded directly from real authentic open-source/Kaggle dataset sources. If data downloading or loading fails, the pipeline fails hard with an explicit exception — it never falls back to dummy or mock data under any circumstances.
