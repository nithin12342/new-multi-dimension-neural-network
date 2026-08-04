---
name: intention-engineering
description: Use this skill whenever the user wants to design or build a nontrivial software system and wants a rigorous, verifiable, low-drift process rather than ad-hoc "vibe coding" or plain context/spec-driven prompting. Intention Engineering fuses Skeleton-of-Thought planning, Domain-Driven Design, OOP/SOLID-shaped file structure, a formal phase state machine with quality gates, traceable node metadata, and hard compiler/execution-verified output (Rust by default). Trigger on requests like "build me a [system/pipeline/service]", "design the architecture for X", "set up the project structure", "plan this out before we code", or any multi-file/multi-module engineering task — even if the user doesn't say "intention engineering" by name. Do NOT use for single-file scripts, one-off snippets, or pure prose/analysis tasks with no code deliverable.
---

# Intention Engineering

An engineering methodology, not just a prompting trick. It differs from **Context Engineering** (stuff relevant context in) and **Spec-Driven Engineering** (write spec, generate code, trust the output) in one respect: nothing is marked done on a claim. Every node of the plan is verified by a compiler and a real execution, every change traces back to the requirement that caused it, and every phase has an explicit exit gate that blocks advancement until satisfied.

This SKILL.md is the front door. It tells you **what layer to read for what you're doing right now** — don't load every reference file for every task; load the one the current step needs (see Context Management below).

## Layered structure

```
Core Principles            -> references/principles.md
        |
State Machine + Gates      -> references/state-machine.md
        |
Execution Algorithm        -> references/execution-algorithm.md
        |
Node Schema + Traceability -> references/node-schema.md
        |
Agile Mapping               -> references/agile-mapping.md
        |
Pattern Library (self-improvement) -> references/pattern-library.md
        |
Language Profiles           -> references/language-profiles/rust.md (default), add siblings for other languages
```

**Principles** are the invariant rules (never violated regardless of project). **Execution Algorithm** is the deterministic input->process->output->validation->failure spec for each phase. **Node Schema** is the data every unit of work must carry. Everything downstream of Principles is mechanical — if you're unsure what to do next, the state machine or the algorithm has the answer, you shouldn't be improvising.

## Roles

- **Planner** (cheap, long-context model): owns the Skeleton document. Never writes implementation logic.
- **Builder** (this instance, or a stronger model): expands exactly one node at a time into verified code.
- **Compiler/Runtime**: the only source of truth. Read `references/language-profiles/rust.md` for the default gate; swap profiles for other languages but never relax the "must pass clean before a node is marked done" rule.

## The pipeline at a glance

```
Planning -> Architecture -> File Design -> Code Skeleton -> Implementation
    -> Compilation -> Execution -> Verification -> Accepted -> Next Node
```

Full state machine, entry/exit criteria per phase, and the compile-fail / verify-fail recovery paths: `references/state-machine.md`. Full input/process/output/validation/failure spec per phase: `references/execution-algorithm.md`.

## Context management (read this before loading anything else into a long session)

Three tiers, never all at once:

- **Permanent context** — this SKILL.md + `principles.md` + the language profile. Loaded once, stays for the whole project.
- **Project context** — the current Skeleton document (`SKELETON.md`), architecture decisions, specs. Reloaded per phase, not per node.
- **Working context** — the current node's metadata, its direct dependencies' signatures, the current compiler/runtime output. Reloaded per node, discarded after.

Budget roughly: first ~20% of usable window for Project context (skeleton subtree only, not the whole tree), last ~20% reserved for compiler/E2E output (never truncate this — it's what drives the next correction), middle for Working context. If a node's true dependency set doesn't fit, the bounded context is too large — split it in the Skeleton, don't cram more in. Use the cheap long-context model for Planning/Architecture; reserve the strong model for Implementation and compiler-error correction.

## Quick-reference checklist per node

- [ ] Node metadata present and current (`references/node-schema.md`)
- [ ] Working context loaded — this node's slice only
- [ ] Signature already exists from Code Skeleton phase, unchanged unless justified
- [ ] Logic implemented
- [ ] If this is a correction: goal stated explicitly, implicated files traced by responsibility, no file touched outside that scope (Correction Scoping Algorithm, `references/execution-algorithm.md`)
- [ ] If this is a correction: structurally-adjacent siblings on the same root-cause chain re-verified, not just this node (Completeness Check, `references/execution-algorithm.md`)
- [ ] Compiler evidence captured, clean or warnings justified
- [ ] Runtime/E2E evidence captured — real input, real output artifact inspected
- [ ] Traceability chain intact (REQ -> SPEC -> SOT -> FOLDER -> FILE -> CLASS/METHOD -> VERIFY)
- [ ] Skeleton updated if this node changed the big picture, propagation algorithm followed
- [ ] Idempotency confirmed if side-effecting
- [ ] Quality gate for this phase satisfied before advancing
- [ ] Committed as one atomic unit with verification evidence attached
- [ ] Pattern-library candidacy assessed (`references/pattern-library.md`)
