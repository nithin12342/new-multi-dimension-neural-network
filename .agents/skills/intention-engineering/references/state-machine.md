# State Machine + Quality Gates

The pipeline is a strict state machine. There is no "skip ahead" transition and no "continue anyway on failure" transition. Every arrow below is the only legal transition from that state.

```
Planning
   | gate: PASS
   v
Architecture
   | gate: PASS
   v
File Design
   | gate: PASS
   v
Code Skeleton
   | gate: PASS
   v
Implementation
   | gate: PASS
   v
Compilation
   | gate: PASS --------- FAIL --> back to Implementation
   v
Execution
   | gate: PASS --------- FAIL --> back to Implementation
   v
Verification
   | gate: PASS --------- FAIL --> back to Implementation (or Rollback, see Recovery)
   v
Accepted
   v
Next Node (loop to Implementation for the next node, or to Planning if the
           bounded context is exhausted, see agile-mapping.md)
```

A phase's gate is checked by the *next* phase's entry criteria — you cannot enter File Design without Architecture's exit criteria having been met, full stop.

## Gate definitions

### Planning -> Architecture
**Exit criteria of Planning:** every bounded context has a name, a one-line reason it's separate, at least one aggregate with a stated invariant, and its ports_in/ports_out declared. `data_flow_order` is populated.
**Entry criteria of Architecture:** the above exists in `SKELETON.md` and has been read.

### Architecture -> File Design
**Exit criteria:** folder tree exists, every top-level folder maps to exactly one DIP layer (`domain/`, `application/`, `infrastructure/`, `interfaces/`), no folder mixes layers.
**Entry criteria:** folder tree committed, no file contents yet.

### File Design -> Code Skeleton
**Exit criteria:** every file has a one-line responsibility statement (<=7 words), its bounded context/aggregate, and what it must never do. SRP check: no two files with the same responsibility statement.
**Entry criteria:** file responsibility list committed.

### Code Skeleton -> Implementation
**Exit criteria:** every file **compiles clean** with all bodies stubbed (`todo!()` / `unimplemented!()` / `raise NotImplementedError`). Imports match the `data_flow_order`, no speculative imports.
**Entry criteria:** skeleton compile evidence attached. If skeleton doesn't compile, return to Architecture or Planning — do not patch it in Code Skeleton.

### Implementation -> Compilation
**Exit criteria:** node's logic written in place of its stub.
**Entry criteria:** node's signature is unchanged from Code Skeleton (or the change is justified and Skeleton updated per the propagation algorithm).

### Compilation gate
**PASS:** clean build, or warnings each individually justified in a one-line comment.
**FAIL:** compiler output captured verbatim -> minimum-diff fix (principle 7) applied -> back to Compilation. Do not proceed to Execution on a failing build under any framing.

### Execution gate
**PASS:** program actually run against the smallest real fixture that exercises this node; output artifact captured.
**FAIL:** back to Implementation. If the failure reveals the fixture itself was wrong, fix the fixture and re-run, but log that as a Skeleton/spec impact, not a silent fixture edit.

### Verification gate
**PASS:** captured output artifact matches the node's stated acceptance criteria (from its node metadata) AND is the exact shape the next phase's consumer expects.
**FAIL:** see Recovery below.

### Accepted -> Next Node
**Exit criteria:** node committed as one atomic unit with compiler evidence + runtime evidence + traceability IDs attached; Skeleton updated if impacted.

## Recovery (failure handling, made explicit)

```
Compile Fail
   -> Correction Scoping Algorithm (execution-algorithm.md): state the goal,
      trace implicated files by responsibility, locate exact function
   -> Fix (minimum diff, strictly within the scoped location)
   -> Compilation (retry)
   -> Execution
   -> Verification
   -> Accept
```

```
Verification Fail
   -> Is the code wrong, or was the acceptance criteria/spec wrong?
        code wrong      -> Correction Scoping Algorithm -> Implementation
                            (minimum diff, scoped) -> Compilation -> Execution -> Verification
        spec/criteria wrong -> update SKELETON.md node metadata (traceable, see node-schema.md)
                              -> re-derive acceptance criteria -> Implementation -> ... -> Verification
   -> If neither resolves after N attempts (team-defined, default 3) -> Rollback:
        revert to last Accepted commit for this node, re-open at Planning
        for this node's parent aggregate — the boundary was likely wrong.
```

Note: a fix that spills outside the Correction Scoping Algorithm's `implicated_files` is not a faster path to green — it is caught automatically by that algorithm's POST-FIX ENFORCEMENT step (`git diff --name-only` checked against `implicated_files`) and hard-reverted before Compilation is retried. This is a mechanical gate, not a self-report — the build passing is never treated as evidence that the diff stayed in scope.

Rollback is a legitimate, expected outcome — not a failure of the process. A boundary that turns out wrong after 3 implementation attempts is exactly the kind of thing Planning-level tools (cheap, long-context model) are cheaper to re-run than repeated Implementation-level attempts (expensive, strong model).
