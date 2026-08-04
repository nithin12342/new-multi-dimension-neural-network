# Execution Algorithm (the "How" — deterministic per phase)

Every phase below follows the same shape: `Input -> Process -> Output -> Validation -> Failure`. Follow this exactly; do not substitute prose instructions for it when actually executing the method.

---

## Phase 0: Planning (domain skeleton)

```
INPUT
  Business goal, requirements (each with a REQ-### id), constraints

PROCESS
  For each requirement, assign it to a bounded context (new or existing)
  For each bounded context, define its aggregates and the invariant each protects
  Define ports_in / ports_out per aggregate (what it consumes/produces, from/to which context)
  Order bounded contexts by data_flow_order (producers before consumers)

OUTPUT
  SKELETON.md populated: bounded_contexts[], data_flow_order[]
  Traceability: REQ-### -> SPEC-### -> SOT-### (one Skeleton-of-Thought node per aggregate)

VALIDATION
  Every REQ-### maps to at least one aggregate
  Every aggregate has a non-empty invariant statement
  data_flow_order is a total order with no cycles

FAILURE
  Cyclic data flow -> the bounded contexts are wrong, re-split them
  Requirement maps to no aggregate -> missing bounded context, add one
```

## Phase 1: Architecture (folder structure)

```
INPUT
  SKELETON.md bounded_contexts[]

PROCESS
  Map each bounded context to domain/<context>/
  Create application/, infrastructure/, interfaces/ per SOLID DIP layering (principles.md #3)
  Create tests/e2e/phaseN_<context>/ per bounded context

OUTPUT
  Folder tree (no file contents)
  Traceability: SOT-### -> FOLDER-###

VALIDATION
  Every folder has: purpose, owning bounded context, DIP layer, no mixed layers
  No folder's purpose statement duplicates another's

FAILURE
  Two folders can't be told apart by purpose -> merge or re-scope in Planning
```

## Phase 2: File Design (responsibilities)

```
INPUT
  Folder tree

PROCESS
  For each folder, enumerate files
  Per file: responsibility (<=7 words), owning aggregate, "must never" clause

OUTPUT
  File responsibility list
  Traceability: FOLDER-### -> FILE-###

VALIDATION
  SRP check: no two files share a responsibility statement
  Every file's responsibility traces to an aggregate in SKELETON.md

FAILURE
  Can't state responsibility in <=7 words -> file is doing too much, split it
```

## Phase 3: Code Skeleton (signatures only)

```
INPUT
  File responsibility list

PROCESS
  Per file, in this exact order:
    1. imports (derived from data_flow_order, nothing speculative)
    2. external package/crate declarations actually needed
    3. struct/class/trait declarations, fields typed, no logic
    4. constructors/destructors, signatures only
    5. trait impls / inheritance declared, bodies stubbed (todo!())
    6. function/method signatures, full types, one-line doc comment

OUTPUT
  Compiling skeleton (see language profile for the exact compile command)
  Traceability: FILE-### -> CLASS-### -> METHOD-###

VALIDATION
  Whole skeleton compiles clean with all bodies stubbed

FAILURE
  Skeleton doesn't compile -> Architecture or Planning was wrong,
  return there. Do NOT patch this in Code Skeleton.
```

## Phase 4: Implementation (one node at a time)

```
INPUT
  One METHOD-### node, its signature, its direct dependencies' signatures,
  its acceptance criteria (from node-schema.md)

PROCESS
  Implement the body
  Compile -> capture evidence
  Run against smallest real fixture -> capture output artifact
  Compare output artifact to acceptance criteria and to next phase's expected
  input shape

OUTPUT
  Verified node: compiler evidence + runtime evidence + traceability
  Traceability: METHOD-### -> VERIFY-###

VALIDATION
  See state-machine.md Compilation/Execution/Verification gates

FAILURE
  See state-machine.md Recovery
```

---

## Correction Scoping Algorithm (runs before every minimum-diff fix)

Mechanical form of principles.md #11. `minimum_diff_fix()` in the node algorithm below is not "find something plausible and patch it" — it always consumes the output of this algorithm first.

```
INPUT
  Failure evidence (compiler error text, or failed verification diff)
  The failing node's acceptance_criteria and traceability chain

PROCESS
  1. State the goal explicitly, one sentence: "this correction must make
     <specific condition> true, and change nothing else."
  2. Walk the traceability chain (node-schema.md) outward from the failing
     node — parent, then the dependency/children edges — collecting every
     file whose stated File Design responsibility plausibly covers the goal.
  3. Narrow: drop any file from that set whose responsibility statement
     does not directly explain the failure. Keep the set as small as it
     can honestly be, not as large as feels safe.
  4. Within each remaining file, locate the exact function/method whose
     responsibility owns the failing behavior — not the file in general.

OUTPUT
  change_plan = { goal: <one sentence>, implicated_files: [...],
                  exact_location: {file -> function/method} }

VALIDATION
  Every entry in implicated_files has a responsibility statement that
  directly explains why it must change for this goal.
  No file outside implicated_files will be touched by the fix.

POST-FIX ENFORCEMENT (hard, automated — not self-reported)
  After minimum_diff_fix() is applied, run:
      actual_changed_files = git diff --name-only
  Compare actual_changed_files to change_plan.implicated_files as sets.
    - actual_changed_files == implicated_files (or a subset)  -> PASS, proceed to Compilation
    - actual_changed_files has ANY entry not in implicated_files -> HARD FAIL:
        git checkout -- <the extra file(s)>   # revert collateral changes immediately
        do NOT proceed to Compilation with the revert unapplied
        re-run Correction Scoping from step 1 if the extra file turns out
        to be genuinely required (i.e. update change_plan deliberately,
        do not just re-allow the diff)
  This check is mechanical and runs every time, regardless of how
  confident the fix felt — principle 11 is only real if this step exists;
  without it, "no collateral damage" is a promise, not a gate.

COMPLETENESS CHECK (runs after the failing node reaches Accepted, before the
                     correction as a whole is declared closed)
  Minimal has a lower bound as well as an upper bound (principles.md #11).
  A diff that stops short of the full causal chain reliably resurfaces as
  a "new" failure elsewhere later — it's the same root cause, not an
  unrelated bug, and must be checked as such before moving on.
      root_cause = change_plan.goal
      candidate_siblings = node.dependencies + node.children
                            # nodes structurally adjacent on the traceability
                            # chain (node-schema.md), not the whole tree
      for sibling in candidate_siblings:
          if sibling.acceptance_criteria could plausibly be violated by
             root_cause (i.e. sibling shares a port, an invariant, or a
             data shape with the node just fixed):
              re-run sibling's existing E2E fixture (do NOT skip this
              because the sibling was previously "done" — status can be
              stale once its dependency changed)
              if sibling's E2E now fails -> it is part of the SAME chain,
                 not a new bug: add it to this correction's change_plan,
                 re-run Correction Scoping for it, repeat until every
                 structurally-adjacent sibling's E2E passes
      only once every checked sibling passes is the correction declared
      complete and the node's skeleton_impact/traceability finalized.

FAILURE
  The goal can't be met without touching a file outside its stated
  responsibility -> the responsibility boundaries are wrong. This is a
  File Design or Architecture issue (state-machine.md), not license to
  touch it anyway. Either return to that phase, or, if urgent, deliberately
  update that file's responsibility statement in the same commit and record
  it as a Skeleton impact (node-schema.md) — never silent scope creep.
  A sibling that keeps failing its E2E after N re-scoping attempts (same
  default of 3 as state-machine.md Recovery) means the root cause was
  mis-identified -> Rollback both nodes together, re-open Planning for
  their shared parent aggregate rather than continuing to chase symptoms.
```

`minimum_diff_fix(result)` then applies the smallest possible edit strictly within `change_plan.exact_location`. If the edit can't be expressed within that location without spilling into another file, that spill is itself evidence the Correction Scoping output was wrong (recompute it) or the boundaries are wrong (see FAILURE above) — it is never resolved by just letting the diff grow. The POST-FIX ENFORCEMENT step above is what actually stops that from happening silently.

## Node execution algorithm (pseudocode)

```
for node in skeleton.nodes_in_data_flow_order():
    if node.status == "done": continue

    working_context = {
        node.metadata,                                  # node-schema.md
        node.file.skeleton_subtree,                       # this aggregate's slice only
        [dep.signature for dep in node.depends_on],
        node.last_compiler_output_if_retry,
    }
    # context budget check (SKILL.md Context Management): if this doesn't
    # fit at ~60% of the effective window, split the bounded context in
    # the Skeleton before continuing -- do not just load less and hope.

    implement(node)
    result = compile()
    node.compiler_evidence = result
    was_a_correction = result.has_errors
    if result.has_errors:
        change_plan = correction_scoping(result, node)     # scoping algorithm above
        patch = minimum_diff_fix(change_plan)               # principles.md #7
        enforce_scope(change_plan)                            # POST-FIX ENFORCEMENT above:
                                                                 # hard-reverts anything outside
                                                                 # change_plan.implicated_files
        retry compile
    if result.has_unjustified_warnings:
        fix_or_annotate(result)

    fixture = smallest_real_input_for(node)
    output = run_program(fixture)
    node.runtime_evidence = output
    assert output.matches(node.acceptance_criteria)
    assert output.shape.matches(downstream_consumer.expected_input_shape)

    if was_a_correction:
        completeness_check(node, change_plan)      # COMPLETENESS CHECK above:
                                                       # re-verify structurally-adjacent
                                                       # siblings before declaring this
                                                       # correction closed, not just this node

    if implementation_revealed_new_invariant_or_port:
        run change_propagation(node)               # below

    node.status = "done"
    node.traceability.verify_id = new_id("VERIFY")
    commit(node)                                    # one ACID unit
    assess_pattern_library_candidacy(node)          # pattern-library.md
```

## Change propagation algorithm

Formalizes "update the Skeleton" into a bounded, auditable operation instead of an open-ended rewrite.

```
Node Changed
   -> Find Parent (the aggregate this node belongs to)
   -> Find Children (nodes that depend on this node's signature/output shape)
   -> Find Dependencies (nodes this node depends on, to check nothing upstream broke)
   -> Determine Impact:
        - signature changed?        -> children must be re-verified, not re-implemented
        - invariant changed?        -> parent aggregate's SKELETON.md entry updated
        - new port opened?          -> data_flow_order re-checked for cycles
        - bounded context too big?  -> split, re-run Planning gate for the new context only
   -> Update SKELETON.md (minimum diff, principles.md #7)
   -> Update Traceability (node-schema.md chain, new VERIFY-### if re-verified)
   -> Regenerate ONLY impacted nodes (never the whole tree)
   -> Reverify each impacted node against its own acceptance criteria
```

This is what makes the loop cheap: a change to one leaf node should touch, at most, its parent aggregate and its direct children — not trigger a full re-plan.
