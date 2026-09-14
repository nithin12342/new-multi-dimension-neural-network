"""One-time purge of checkpoints trained with the buggy manifold structure.

Run ONCE in Colab BEFORE resuming training:

    !python purge_corrupted_checkpoints.py --confirm

This deletes every stale weight file so the pipeline finds no checkpoint to
resume from and restarts cleanly from epoch 1 / data sample 1:

  - <base>/checkpoints/model_01..model_06/**/* (.safetensors, .pt, .ckpt)
  - <base>/dummy_weights/**/* (dummy_v1.safetensors)
  - ~/.cache/local_checkpoints/**/* (latest_local.pt)

One-time guard: after a successful purge a marker file is written to
<base>/logs/.corrupted_weights_purged_once and any re-run refuses to do
anything unless --force is passed. The script itself is kept in the repo
as a record; it must never be wired into the training pipeline.

Stdlib only. Never touches datasets, logs, telemetry, or source code.
"""

import argparse
import os
import sys

WEIGHT_EXTENSIONS = (".safetensors", ".pt", ".ckpt")
MARKER_NAME = ".corrupted_weights_purged_once"

CANDIDATE_BASES = [
    "/content/drive/MyDrive/SOTA_Cluster_Shared",
    "/content/SOTA_Cluster_Shared",
    os.path.abspath("./SOTA_Cluster_Shared"),
]


def resolve_base(explicit: str | None) -> str:
    if explicit:
        return os.path.abspath(explicit)
    for candidate in CANDIDATE_BASES:
        if os.path.isdir(candidate):
            return candidate
    return CANDIDATE_BASES[0]


def collect_weight_files(base: str) -> list[str]:
    targets: list[str] = []
    for category in ("checkpoints", "dummy_weights"):
        root = os.path.join(base, category)
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.lower().endswith(WEIGHT_EXTENSIONS):
                    targets.append(os.path.join(dirpath, name))
    local_cache = os.path.join(os.path.expanduser("~"), ".cache", "local_checkpoints")
    if os.path.isdir(local_cache):
        for dirpath, _, filenames in os.walk(local_cache):
            for name in filenames:
                if name.lower().endswith(WEIGHT_EXTENSIONS):
                    targets.append(os.path.join(dirpath, name))
    return sorted(targets)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-time deletion of checkpoints trained with the buggy manifold structure."
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Cluster shared base dir (default: auto-detect).",
    )
    parser.add_argument(
        "--confirm", action="store_true", help="Required: actually delete the files."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override the one-time marker and run again.",
    )
    parser.add_argument(
        "--local-cache-only",
        action="store_true",
        help="Only purge ~/.cache/local_checkpoints.",
    )
    args = parser.parse_args()

    base = resolve_base(args.base_dir)
    marker = os.path.join(base, "logs", MARKER_NAME)

    if os.path.exists(marker) and not args.force:
        print(
            f"[PURGE] One-time purge already completed (marker: {marker}). Nothing to do. Kept as record.",
            flush=True,
        )
        return 0

    if args.local_cache_only:
        files = [f for f in collect_weight_files(base) if ".cache" in f]
    else:
        files = collect_weight_files(base)

    print(f"[PURGE] Base dir: {base}", flush=True)
    print(f"[PURGE] Stale weight files found: {len(files)}", flush=True)
    for path in files[:50]:
        print(f"        {path}", flush=True)
    if len(files) > 50:
        print(f"        ... and {len(files) - 50} more", flush=True)

    if not args.confirm:
        print(
            "[PURGE] DRY RUN — nothing deleted. Re-run with --confirm to delete.",
            flush=True,
        )
        return 0

    failures: list[str] = []
    for path in files:
        try:
            os.remove(path)
        except Exception as exc:  # noqa: BLE001 — report and continue
            failures.append(f"{path}: {exc}")

    try:
        os.makedirs(os.path.join(base, "logs"), exist_ok=True)
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write(
                f"purged {len(files) - len(failures)} stale weight files; failures={len(failures)}\n"
            )
    except Exception as exc:  # noqa: BLE001 — marker is best-effort
        print(f"[PURGE] WARNING: could not write one-time marker: {exc}", flush=True)

    if failures:
        print(
            f"[PURGE] Deleted {len(files) - len(failures)}/{len(files)}; FAILURES:",
            flush=True,
        )
        for entry in failures:
            print(f"        {entry}", flush=True)
        return 1

    print(
        f"[PURGE] Deleted {len(files)} stale weight files. Next pipeline run starts from epoch 1 / sample 1.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
