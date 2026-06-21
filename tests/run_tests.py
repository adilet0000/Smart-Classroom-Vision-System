"""
Zero-dependency test runner.

pytest is not a hard dependency of this project, so this runner lets the sanity
tests run with a plain interpreter:

    python tests/run_tests.py          # run every test module
    python tests/test_scoring.py       # run a single module

If pytest *is* installed, `pytest tests/` works too — the test functions are
ordinary `test_*` functions.
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback
from types import ModuleType

# Make the repo root importable (so `src` and `tests` packages resolve) whether
# this is run as `python tests/run_tests.py` or `python -m tests.run_tests`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

TEST_MODULES = [
    "tests.test_geometry",
    "tests.test_scoring",
    "tests.test_temporal",
    "tests.test_embedding_cache",
]


def run_module(module: ModuleType) -> bool:
    """Run every ``test_*`` callable in *module*. Returns True if all passed."""
    funcs = sorted(
        name for name in dir(module)
        if name.startswith("test_") and callable(getattr(module, name))
    )
    passed = 0
    failed = 0
    for name in funcs:
        try:
            getattr(module, name)()
            print(f"  PASS  {module.__name__}.{name}")
            passed += 1
        except Exception:  # noqa: BLE001 — report any failure
            failed += 1
            print(f"  FAIL  {module.__name__}.{name}")
            traceback.print_exc()
    print(f"  -> {passed} passed, {failed} failed in {module.__name__}\n")
    return failed == 0


def main() -> int:
    ok = True
    for mod_name in TEST_MODULES:
        module = importlib.import_module(mod_name)
        ok = run_module(module) and ok
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
