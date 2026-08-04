# Language Profile: Rust (default)

Rust is default because the borrow checker + exhaustive `match` + strong types reject entire classes of half-finished nodes at compile time — this is a feature of the verification gate (state-machine.md Compilation), not a style preference.

## Compile commands per phase

- Code Skeleton gate: `cargo check` — fast, type/borrow errors only, no codegen needed for stubbed bodies.
- Implementation gate: `cargo build` — full build, catches anything `check` might defer.
- Lint: `cargo clippy -- -D warnings` — run before a node is marked done; a clippy warning is either fixed or justified with a one-line `#[allow(...)]` comment stating why.

## Stub convention (Code Skeleton phase)

```rust
todo!()            // for functions that will panic-if-reached during skeleton compile
unimplemented!()    // equivalent, use whichever reads better for the file
```

Never use a fake return value (`Ok(Default::default())` etc.) as a stub — it compiles but silently passes the Execution gate with wrong output, defeating the verification model.

## Constructors/destructors

- Constructors: `fn new(...) -> Self` or `fn from_x(...) -> Result<Self, E>` — prefer `Result` for anything that can fail (file I/O, parsing), matching the invariant it's protecting (principles.md #2, #9 Consistency).
- Destructors: implement `Drop` only when the struct owns a resource the type system doesn't already clean up (raw handles, temp files, locks) — signature-only in Code Skeleton, real cleanup logic in Implementation.

## Ownership and DIP layering

`domain/` types should be plain data + pure functions with no `Rc<RefCell<>>` unless the invariant genuinely requires shared mutable state — reach for that as a last resort, since it usually signals a bounded-context boundary was drawn wrong (state-machine.md Recovery: rollback and re-plan rather than reach for interior mutability to paper over it).

## E2E test harness convention

```
tests/e2e/phaseN_<context>/
├── fixtures/    # real input files
├── expected/    # the artifact the NEXT phase's program must be able to read
└── run.sh       # idempotent: re-running produces identical expected/ output
```

`run.sh` invokes the actual built binary (`cargo run --bin <interface>`), not a Rust `#[test]` function — the point is exercising the real compiled artifact exactly as it will run in production, not an in-process test harness.

## ACID / commit mapping

One node = one commit. Commit message template:

```
[METHOD-021] <one line: what this node now does>

Compiler evidence: cargo build — 0 errors, 0 warnings
Runtime evidence: tests/e2e/phase1_ingestion/expected/sample_output.txt
Traceability: REQ-001 -> SPEC-001 -> SOT-001 -> FILE-001 -> METHOD-021 -> VERIFY-021
Skeleton impact: none
```

## Adapting to another language

Keep the gate shape identical, swap only the tool:

| Rust | Equivalent slot |
|---|---|
| `cargo check` | strictest type-checker in strict/pedantic mode |
| `cargo build` | full compile/build step |
| `cargo clippy -D warnings` | strictest linter available, warnings-as-errors |
| `todo!()`/`unimplemented!()` | language's explicit "not implemented" raise |
| `Result<T, E>` | language's explicit error type, not exceptions-as-control-flow where avoidable |

Never relax "must pass clean before a node is marked done" when swapping profiles — that rule is a Principle (principles.md #4), not a Rust-specific convenience. Add a new file under `references/language-profiles/<language>.md` following this same shape rather than diluting this one into a multi-language document.
