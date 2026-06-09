import importlib.util
import inspect
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    failures = []
    for path in sorted(ROOT.glob("test_*.py")):
        module = load_module(path)
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            try:
                func()
                print(f"PASS {path.name}::{name}")
            except Exception as exc:
                failures.append((path.name, name, exc))
                print(f"FAIL {path.name}::{name}: {exc}")

    if failures:
        raise SystemExit(1)

    print("All tests passed.")


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
