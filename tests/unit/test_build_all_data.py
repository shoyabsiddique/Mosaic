"""build_all_data.py must not let one source's failure (e.g. CFPB rate-limiting after
repeated hits -- confirmed to actually happen mid-development) prevent the other
domains from building. `scripts/` isn't an importable package, so this loads the
script directly from its file path.
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "build_all_data.py"


def _load_module_without_running_main():
    spec = importlib.util.spec_from_file_location("build_all_data_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # importing runs top-level code, including the `from decision_engine... import`
    # lines and the REAL_LOADERS/SYNTHETIC_GENERATORS list construction, but not
    # main() itself (guarded by `if __name__ == "__main__"`, false for an imported module)
    spec.loader.exec_module(module)
    return module


def _fake_module(name: str, n_records: int = 0, error: Exception = None):
    def build_and_save(out_path):
        if error is not None:
            raise error
        return n_records
    return SimpleNamespace(__name__=name, build_and_save=build_and_save)


def test_run_all_continues_past_a_failing_source_and_reports_it(tmp_path):
    build_all_data = _load_module_without_running_main()

    ok_module = _fake_module("ok_source", n_records=5)
    failing_module = _fake_module("failing_source", error=RuntimeError("simulated 403"))
    another_ok_module = _fake_module("another_ok_source", n_records=3)

    entries = [
        (failing_module, tmp_path / "a.jsonl"),
        (ok_module, tmp_path / "b.jsonl"),
        (another_ok_module, tmp_path / "c.jsonl"),
    ]

    total, succeeded, failed = build_all_data._run_all(entries, "test")

    assert total == 8  # 5 + 3, the failing one contributes 0, not a crash
    assert succeeded == ["ok_source", "another_ok_source"]
    assert failed == ["failing_source"]


def test_run_all_with_no_failures_reports_empty_failed_list(tmp_path):
    build_all_data = _load_module_without_running_main()
    entries = [(_fake_module("a", n_records=1), tmp_path / "a.jsonl")]

    total, succeeded, failed = build_all_data._run_all(entries, "test")

    assert total == 1
    assert succeeded == ["a"]
    assert failed == []


def test_run_all_with_all_failures_does_not_raise(tmp_path):
    build_all_data = _load_module_without_running_main()
    entries = [
        (_fake_module("a", error=ValueError("boom")), tmp_path / "a.jsonl"),
        (_fake_module("b", error=ConnectionError("boom2")), tmp_path / "b.jsonl"),
    ]

    total, succeeded, failed = build_all_data._run_all(entries, "test")

    assert total == 0
    assert succeeded == []
    assert failed == ["a", "b"]
