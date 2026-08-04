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
Before touching any code to fix an error, state the goal of the correction explicitly: the exact failure it must resolve, in terms of the node's `acceptance_criteria` or the compiler error, not a vague "make it work." Then trace the chain of implicated files using their stated responsibilities (Phase 2, File Design) and the traceability chain (`node-schema.md`) — do not guess which files are relevant, derive them from the chain. Within that traced set, locate the exact function/method the failure's responsibility belongs to. Change only that location.

A fix that touches a file, function, or line outside the traced chain is collateral damage. This is not enforced by self-discipline: `execution-algorithm.md`'s Correction Scoping Algorithm ends with a POST-FIX ENFORCEMENT step that runs `git diff --name-only` against the traced file set after every fix and hard-reverts anything outside it, automatically, before the next Compilation attempt — the same way a compiler error blocks progress regardless of how confident the fix felt. If the goal genuinely cannot be met without touching something outside the current responsibility boundaries, that is a signal the boundaries themselves are wrong: this is a File Design or Architecture issue (state-machine.md), not license to expand the diff. Either return to that phase properly, or — if urgent — deliberately update the affected file's responsibility statement in the same commit and log it as a Skeleton impact (`node-schema.md`), never as a silent scope creep.

Minimal has a lower bound as well as an upper bound, and the lower bound is just as often violated: a diff that is small but stops short of the full causal chain is not minimal, it's incomplete, and it reliably manifests as a symptom reappearing somewhere else after the "fix" is accepted — this is a chained-failure pattern, not an independent-bugs pattern, and treating each failure as independent is what produces it. Fixing node A's compile error while leaving node B (which shares A's root cause via a `dependencies`/`children` edge) unrepaired regularly resurfaces as B's E2E gate failing later, mistaken for a new, unrelated bug. Correction Scoping therefore does not close on a single node's green gate — it re-checks the sibling nodes on the same root-cause chain before the correction is declared complete. See `execution-algorithm.md`'s COMPLETENESS CHECK step for the mechanical version. Minimal, in this method, means: **the smallest diff that resolves the entire causal chain, not the smallest diff that resolves the symptom in front of you.**

See `execution-algorithm.md`'s Correction Scoping Algorithm for the mechanical version of this principle; it runs before every application of principle 7's minimum-diff fix.
