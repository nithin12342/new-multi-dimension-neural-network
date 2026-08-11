# Intention Engineering Core Principles & Non-Negotiable System Invariants

## Core Principles

1. **Systematic Design:** Always follow Intention Engineering workflow (Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 -> Phase 4).
2. **Master Blueprint:** `SKELETON.md` is the living single source of truth blueprint.
3. **Single Responsibility Principle (SRP):** Every Python file must have a single responsibility defined in $\le 7$ words and an explicit `Must Never` clause.
4. **Dependency Inversion Principle (DIP):** Domain models under `src/domain/` MUST NEVER import from `src/infrastructure/` or `src/application/`.

## Mandatory Invariant Rules

- **Rule 12 (Authentic Data Only):**
  No synthetic/mock data generation, no dummy tensor fallbacks, and no mock fallouts. Authentic open-source datasets (torchvision datasets, Kaggle key `KGAT_c0234fe2d5a9d53f6c18baf6fbe983b4`, Encord `encord-team/E-MM1-1M`) used exclusively.

- **Rule 13 (Strict Local Execution Boundary):**
  DO NOT execute full training loops or pretraining runs on the local developer PC. Local execution of 6 CUDA streams / heavy model loops causes local system crashes/freezes. Training execution is STRICTLY RESTRICTED to Google Colab GPU cloud environments (`python -m src.interfaces.cli.main` in Colab). Local commands are used strictly for code modification, unit testing/dry-runs, and git syncing.
