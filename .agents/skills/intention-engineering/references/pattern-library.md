# Pattern Library (the self-improving final stage)

The method as described so far ends at "Verified Node -> Next Node." That's incomplete: a project that never extracts what it learns re-derives the same solutions from scratch every time, which wastes exactly the verification effort the method just spent proving something correct. Add a final stage:

```
Verified Node
   |
Documentation        (what this node does and why, in terms of its acceptance
                       criteria and invariant — not a restatement of the code)
   |
Skill Extraction      (is the solution shape reusable independent of this
                       project's specific domain? e.g. "idempotent file-write
                       adapter" is reusable; "RawDocument invariant" usually isn't)
   |
Knowledge Base Update (add to references/pattern-library-entries/ or the
                       project's own pattern store, with its traceability
                       chain preserved so provenance is never lost)
   |
Pattern Library        (a growing set of pre-verified building blocks)
   |
Next Node               (future Planning phases check the pattern library
                        BEFORE generating a new node from scratch)
```

## When to promote a node to a pattern

Ask at the "assess_pattern_library_candidacy" step (execution-algorithm.md node algorithm):

- Would this node's shape (not its specific domain types) recur in a different bounded context, or a different project entirely?
- Is its acceptance criteria expressible generically (e.g. "idempotent upsert given a key + payload") rather than tied to this project's nouns?
- Did it take more than one Compilation/Verification retry to get right? (High-cost-to-derive solutions are the highest-value ones to not re-derive.)

If yes to any: add an entry to `pattern_library_candidates` in `SKELETON.md` (node-schema.md schema), with the node's traceability chain attached so the pattern is never divorced from where it was proven correct.

## Pattern library entry format

```yaml
pattern_id: PATTERN-###
name: <short name>
generalized_acceptance_criteria: <domain-independent statement>
source_node: <METHOD-### this was extracted from, traceability preserved>
applicable_when: <what conditions in a new Skeleton make this pattern relevant>
language_profile: <which language-profiles/*.md it was verified under>
verification_evidence_ref: <VERIFY-### — reuse doesn't waive re-verification
                             in the new context, but it shortcuts Planning
                             and Code Skeleton>
```

## Why this closes the loop

Without this stage, "self-improving" is a marketing word. With it, every sprint retro (agile-mapping.md) has a concrete question — "did we add to the pattern library, and did we check it before planning this sprint's new nodes?" — that makes the improvement measurable instead of aspirational.
