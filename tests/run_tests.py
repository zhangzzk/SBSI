"""Run the SBSI unit tests without pytest.

`pytest` is not installed in the `sims1` env, and only some test modules carry a
`__main__` guard. This runner imports every `tests/test_*.py`, calls each top-level
`test_*` function, and reports pass/fail counts.

    python tests/run_tests.py            # all modules
    python tests/run_tests.py shear_map  # only matching modules
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import traceback

TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> int:
    patterns = argv[1:]
    passed = failed = 0
    for path in sorted(TESTS.glob("test_*.py")):
        if patterns and not any(p in path.stem for p in patterns):
            continue
        try:
            module = load(path)
        except Exception:
            print(f"{path.name}: IMPORT FAILED")
            traceback.print_exc()
            failed += 1
            continue
        names = [n for n in dir(module) if n.startswith("test_") and callable(getattr(module, n))]
        mod_failed = 0
        for name in names:
            try:
                getattr(module, name)()
            except Exception:
                mod_failed += 1
                print(f"{path.name}::{name} FAILED")
                traceback.print_exc(limit=3)
        passed += len(names) - mod_failed
        failed += mod_failed
        print(f"{path.name}: {len(names) - mod_failed}/{len(names)} passed")

    print(f"\ntotal: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
