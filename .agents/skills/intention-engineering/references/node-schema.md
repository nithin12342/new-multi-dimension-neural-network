# Node Metadata + Traceability

## Node metadata (every node carries all of this)

```yaml
node_id: <METHOD-###>            # unique, see traceability chain below
parent: <aggregate/file id this belongs to>
children: [<node ids that depend on this node's output/signature>]
dependencies: [<node ids this node depends on>]
priority: <critical | high | normal | low>
status: stub | building | compiled | e2e_passed | verified | done
risk: <low | medium | high>       # likelihood this reveals a wrong boundary
complexity: <trivial | moderate | complex>
owner: <planner | builder | human>   # who is accountable for this node right now
acceptance_criteria: <what the output artifact must satisfy, stated concretely>
verification_evidence:
  compiler_evidence: <verbatim output, last error/warning block if long>
  verification_command: <the exact command run to verify this node, e.g.
                          "cargo run --bin ingest -- tests/e2e/phase1/fixtures/sample.pdf">
  expected_output: <what a human can compare the actual output against
                     without trusting the model's claim, e.g. a path to
                     tests/e2e/phase1/expected/sample_output.json, or an
                     explicit value>
  runtime_evidence: <path to captured output artifact actually produced>
completeness_check:                 # populated only when this node was a correction
  root_cause: <change_plan.goal from the correction that touched this node>
  siblings_checked: [<node ids re-verified on the same chain>]
  siblings_status: <all_passed | reopened: [<node ids that failed and were
                     folded back into the correction>]>
traceability:
  req_id: <REQ-###>
  spec_id: <SPEC-###>
  sot_id: <SOT-###>
  folder_id: <FOLDER-###>
  file_id: <FILE-###>
  class_id: <CLASS-###>
  method_id: <METHOD-###>
  verify_id: <VERIFY-###>
skeleton_impact: none | updated_invariant | new_port | split_context
```

`owner` matters once more than one model/human is working the tree in parallel — it prevents two agents from both claiming the same node, which is the isolation property (principles.md #9) made concrete at the node level.

## Traceability ID chain

```
REQ-001  (a stated requirement)
   |
SPEC-001  (the spec derived from it)
   |
SOT-005   (the Skeleton-of-Thought / aggregate node covering it)
   |
FOLDER-02 (the folder that houses its bounded context)
   |
FILE-014  (the file responsible for it)
   |
CLASS-007 (the struct/class/trait)
   |
METHOD-021 (the specific function/method)
   |
VERIFY-021 (the verification record: compiler + runtime evidence)
```

Every ID is assigned once and never reused. Break the chain and change-impact analysis (execution-algorithm.md's propagation algorithm) has nothing to walk — this is not optional bookkeeping, it's the mechanism that makes "regenerate only impacted nodes" possible instead of "regenerate everything to be safe."

## Full SKELETON.md schema

```yaml
system: <name>
requirements:
  - id: REQ-001
    text: <one line>
    spec_id: SPEC-001

bounded_contexts:
  - name: <BoundedContext>
    reason_separate: <one line>
    sot_id: SOT-###
    aggregates:
      - name: <Aggregate>
        invariant: <what must always hold true>
        entities: [<Entity>, ...]
        value_objects: [<VO>, ...]
        ports_in: [<what it consumes, and from which context>]
        ports_out: [<what it produces, and for which context>]
    files:
      - path: domain/<aggregate>.rs
        file_id: FILE-###
        folder_id: FOLDER-###
        responsibility: <=7 words>
        status: skeleton | in_progress | verified
        depends_on: [<file ids>]

data_flow_order: [ContextA, ContextB, ContextC]

nodes:
  # one entry per node-metadata block above, keyed by node_id

pattern_library_candidates:
  - node_id: <METHOD-###>
    reason: <why this generalizes beyond this project>
```

## Worked micro-example (Rust, 2 bounded contexts, with traceability)

```yaml
requirements:
  - id: REQ-001
    text: "Parse uploaded PDF into structured document"
    spec_id: SPEC-001
  - id: REQ-002
    text: "Validate structured document against rule set"
    spec_id: SPEC-002

bounded_contexts:
  - name: Ingestion
    reason_separate: "owns raw file parsing, nothing else touches raw bytes"
    sot_id: SOT-001
    aggregates:
      - name: RawDocument
        invariant: "bytes are never mutated after ingestion; only read"
        ports_out: ["ParsedDocument -> Validation context"]
    files:
      - path: domain/ingestion/raw_document.rs
        file_id: FILE-001
        folder_id: FOLDER-001
        responsibility: "parse raw bytes into RawDocument"
        depends_on: []

  - name: Validation
    reason_separate: "owns rule evaluation, never touches raw bytes"
    sot_id: SOT-002
    aggregates:
      - name: RuleCheck
        invariant: "a RuleCheck result references exactly one ParsedDocument"
        ports_in: ["ParsedDocument <- Ingestion context"]
    files:
      - path: domain/validation/rule_check.rs
        file_id: FILE-002
        folder_id: FOLDER-002
        responsibility: "evaluate rules against ParsedDocument"
        depends_on: [FILE-001]

data_flow_order: [Ingestion, Validation]

nodes:
  - node_id: METHOD-001
    parent: FILE-001
    children: [METHOD-002]
    dependencies: []
    priority: critical
    status: done
    risk: low
    complexity: moderate
    owner: builder
    acceptance_criteria: "given a real sample PDF, returns RawDocument with byte count matching file size"
    verification_evidence:
      compiler_evidence: "cargo build: 0 errors, 0 warnings"
      runtime_evidence: tests/e2e/phase1_ingestion/expected/sample_output.txt
    traceability: {req_id: REQ-001, spec_id: SPEC-001, sot_id: SOT-001,
                    folder_id: FOLDER-001, file_id: FILE-001,
                    class_id: CLASS-001, method_id: METHOD-001,
                    verify_id: VERIFY-001}
    skeleton_impact: none
```

Phase-3 skeleton for `FILE-001` (must compile with stubbed body before Implementation starts):

```rust
// domain/ingestion/raw_document.rs — owns raw parsed bytes, read-only after ingest
pub struct RawDocument {
    pub path: std::path::PathBuf,
    pub bytes: Vec<u8>,
}

impl RawDocument {
    pub fn from_path(path: std::path::PathBuf) -> Result<Self, IngestError> {
        todo!()
    }
}
```

## Testing note (why E2E-only, not unit tests)

"Verified" here means: the actual compiled program ran against a real input file and produced the actual artifact the next phase needs to consume — not a mocked collaborator satisfying an isolated function-level assertion. `tests/e2e/phaseN_<context>/fixtures/` holds real inputs, `tests/e2e/phaseN_<context>/expected/` holds the artifact the next phase must be able to read. The chain across phases is itself the integration test; there is no separate unit-test layer to maintain or let drift from the real interfaces.
