# Agile Mapping

Not "one node = one increment" alone — a full hierarchy, so the method plugs into existing agile ceremonies without inventing new vocabulary for the same things.

```
Epic
  = one bounded context (SKELETON.md bounded_contexts[])
   |
Feature
  = one aggregate within that context
   |
Skeleton Branch
  = the git branch/worktree where that aggregate's nodes are being built
    (principles.md #9, Isolation)
   |
Node
  = one METHOD-### (node-schema.md)
   |
Implementation
  = execution-algorithm.md Phase 4 for that node
   |
Verification
  = state-machine.md Compilation/Execution/Verification gates
   |
Done
  = Accepted state, committed, traceability chain closed
```

## Sprint definition

One sprint = one full Phase-4 pass over one Feature (aggregate) — every node in that aggregate reaches `done`. Sprint length is however long that takes; this method doesn't fix a calendar duration, it fixes a *unit of completion*: a sprint never ends with nodes half-implemented and untested, because the E2E gate is the definition of "done," not a separate QA step tacked on after.

## Retro

A real retro, not ceremony: at sprint end, diff `SKELETON.md` against what was actually learned during Implementation (`skeleton_impact` fields on the sprint's nodes). Every `updated_invariant`, `new_port`, or `split_context` entry is retro material — it's a place the plan was wrong and got corrected, which is exactly what a retro should surface. Also review `pattern_library_candidates` added this sprint (pattern-library.md) — did any node's solution turn out to be reusable beyond this feature?

## Shipping cadence

Ship after every `done` node if the interface allows it — increments should be small enough to actually be shippable, not just "the plan says so." A node that can't be shipped alone (e.g. it's mid-aggregate and the aggregate's invariant isn't yet fully enforced) is a signal the aggregate boundary might be too coarse; consider whether it should be split into two aggregates with independently-shippable invariants.

## Planning cadence

Planning (Phase 0) re-runs at Epic granularity when a new bounded context is needed, and at Feature granularity (a lighter re-plan) when Rollback (state-machine.md Recovery) sends a Feature back for re-scoping. It does not re-run for every node — that would defeat the point of having a Skeleton at all.
