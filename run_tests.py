#!/usr/bin/env python3
"""
FILE: run_tests.py (Unified Multi-Language Test Runner)
Executes:
  1. Rust test suite (cargo test) across crates/nfm-core
  2. Python unit test suite (unittest discover) across tests/unit/
"""

import sys
import subprocess
import os

def run_command(cmd, desc):
    print(f"\n=======================================================", flush=True)
    print(f"  RUNNING: {desc}", flush=True)
    print(f"  COMMAND: {' '.join(cmd)}", flush=True)
    print(f"=======================================================\n", flush=True)
    res = subprocess.run(cmd, shell=False)
    if res.returncode != 0:
        print(f"\n[FAILED] {desc} returned exit code {res.returncode}", flush=True)
        return False
    print(f"\n[PASSED] {desc} completed successfully!", flush=True)
    return True

def main():
    root_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(root_dir)

    success = True

    # 1. Run Cargo tests at workspace root
    success &= run_command(["cargo", "test"], "Rust Native Runtime Test Suite (crates/nfm-core)")

    # 2. Run Python tests
    python_exe = sys.executable
    success &= run_command(
        [python_exe, "-m", "unittest", "discover", "-s", "tests/unit", "-p", "test_*.py"],
        "Python Comprehensive Unit Test Suite (tests/unit/)"
    )

    if not success:
        print(f"\n[SUMMARY] Some test suites failed.", flush=True)
        return 1

    print(f"\n=======================================================", flush=True)
    print(f"  ALL TEST SUITES PASSED CLEANLY ACROSS RUST & PYTHON!", flush=True)
    print(f"=======================================================\n", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
