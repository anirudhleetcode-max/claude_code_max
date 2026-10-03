from __future__ import annotations

import pytest

from ai_engineer.tester.models import CheckKind, CheckResult, Diagnostic, TestCaseFailure, normalize_text
from ai_engineer.tester.parsers import MAX_ITEMS, OUTPUT_TAIL_CHARS, parse_output

T, L, F, TY, B, A = CheckKind.TEST, CheckKind.LINT, CheckKind.FORMAT, CheckKind.TYPECHECK, CheckKind.BUILD, CheckKind.AUDIT

# ---------------------------------------------------------------------------------------------- pytest

PYTEST_FAILURES = """\
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/dev/proj
collected 3 items

tests/test_x.py .FF                                                      [100%]

=================================== FAILURES ===================================
___________________________________ test_bad ___________________________________

    def test_bad():
>       assert 1 == 2
E       assert 1 == 2

tests/test_x.py:7: AssertionError
_________________________________ TestA.test_b _________________________________

self = <test_x.TestA object at 0x7f1f0936b390>

    def test_b(self):
>       compute()

tests/test_x.py:11:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    def compute():
>       raise ValueError("boom at 0x7f3a")
E       ValueError: boom at 0x7f3a

src/calc.py:3: ValueError
=========================== short test summary info ============================
FAILED tests/test_x.py::test_bad - assert 1 == 2
FAILED tests/test_x.py::TestA::test_b - ValueError: boom at 0x7f3a
========================= 2 failed, 1 passed in 0.03s ==========================
"""


def test_pytest_failures_with_sections_and_short_summary() -> None:
    r = parse_output(T, "python -m pytest", PYTEST_FAILURES, 1, duration_s=12.34)
    assert r.status == "failed" and not r.ok()
    assert r.classification == "test_failure"
    assert (r.passed, r.failed, r.errors, r.skipped) == (1, 2, 0, 0)
    assert r.summary == "pytest: 2 failed, 1 passed (12.3s)"
    bad, b = r.failures
    assert bad.test_id == "tests/test_x.py::test_bad"
    assert (bad.file, bad.line, bad.failure_type) == ("tests/test_x.py", 7, "assertion")
    assert bad.message == "assert 1 == 2"
    assert b.test_id == "tests/test_x.py::TestA::test_b"
    assert (b.file, b.line) == ("src/calc.py", 3)  # where the exception was raised
    assert b.failure_type == "error"
    assert r.output_tail.endswith("==========================\n")


def test_pytest_bracketed_summary_with_all_counts() -> None:
    out = """\
FAILED tests/test_a.py::test_one - AssertionError: assert 'a' == 'b'
FAILED tests/test_a.py::test_two[1-2] - TypeError: unsupported operand type(s)
ERROR tests/test_b.py::test_fixture - RuntimeError: db down
=========== 2 failed, 10 passed, 1 skipped, 1 error, 3 warnings in 1.23s ===========
"""
    r = parse_output(T, "pytest -q", out, 1)
    assert (r.passed, r.failed, r.errors, r.skipped) == (10, 2, 1, 1)
    assert [f.failure_type for f in r.failures] == ["assertion", "type", "error"]
    assert r.failures[1].test_id == "tests/test_a.py::test_two[1-2]"
    assert r.classification == "test_failure"
    assert r.summary == "pytest: 2 failed, 1 error, 10 passed, 1 skipped"


def test_pytest_quiet_style_summary() -> None:
    out = "..F.\nFAILED tests/test_q.py::test_x - assert 0\n2 failed, 10 passed in 1.2s\n"
    r = parse_output(T, "python -m pytest -q", out, 1)
    assert (r.failed, r.passed) == (2, 10)
    assert r.failures[0].test_id == "tests/test_q.py::test_x"


def test_pytest_all_passed() -> None:
    out = "........\n============================== 8 passed in 0.52s ===============================\n"
    r = parse_output(T, "python -m pytest", out, 0, duration_s=0.52)
    assert r.ok() and r.status == "passed" and r.classification == "passed"
    assert r.passed == 8 and r.failed == 0
    assert r.summary == "pytest: 8 passed (0.5s)"
    assert r.signature() == ""


PYTEST_COLLECTION_Q = """
==================================== ERRORS ====================================
_______________________ ERROR collecting tests/test_y.py _______________________
ImportError while importing test module '/home/dev/proj/tests/test_y.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/usr/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/test_y.py:1: in <module>
    import requests_mock
E   ModuleNotFoundError: No module named 'requests_mock'
=========================== short test summary info ============================
ERROR tests/test_y.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.09s
"""


def test_pytest_collection_error_missing_dependency() -> None:
    r = parse_output(T, "python -m pytest -q", PYTEST_COLLECTION_Q, 2)
    assert r.classification == "missing_dependency"
    assert r.status == "error"
    (f,) = r.failures
    assert f.test_id == "tests/test_y.py"
    assert (f.file, f.line) == ("tests/test_y.py", 1)
    assert f.message == "ModuleNotFoundError: No module named 'requests_mock'"
    assert f.failure_type == "import"
    assert r.errors == 1
    assert "requests_mock" in r.summary


def test_pytest_collection_error_with_message_in_short_summary_and_local_module() -> None:
    out = """\
_____________________ ERROR collecting tests/test_api.py ______________________
tests/test_api.py:3: in <module>
    from myapp.api import handler
E   ModuleNotFoundError: No module named 'myapp.api'
=========================== short test summary info ============================
ERROR tests/test_api.py - ModuleNotFoundError: No module named 'myapp.api'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 0.12s ===============================
see src/myapp/__init__.py
"""
    r = parse_output(T, "pytest", out, 2)
    assert r.classification == "import_error"  # module path is local to the project
    assert r.status == "failed"


def test_pytest_collection_import_name_error_is_import_error() -> None:
    out = """\
_____________________ ERROR collecting tests/test_calc.py _____________________
tests/test_calc.py:1: in <module>
    from calc import multiply
E   ImportError: cannot import name 'multiply' from 'calc' (/home/dev/proj/calc.py)
=========================== short test summary info ============================
ERROR tests/test_calc.py
=============================== 1 error in 0.05s ===============================
"""
    r = parse_output(T, "pytest", out, 2)
    assert r.classification == "import_error"
    assert r.failures[0].failure_type == "import"


def test_pytest_collection_syntax_error() -> None:
    out = """\
==================================== ERRORS ====================================
_____________________ ERROR collecting tests/test_z.py ________________________
/venv/lib/python3.11/site-packages/_pytest/python.py:498: in importtestmodule
    mod = import_path(
E     File "/home/dev/proj/tests/test_z.py", line 3
E       def broken(:
E                  ^
E   SyntaxError: '(' was never closed
=========================== short test summary info ============================
ERROR tests/test_z.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 0.10s ===============================
"""
    r = parse_output(T, "pytest", out, 2)
    assert r.classification == "syntax_error"
    f = r.failures[0]
    assert f.message == "SyntaxError: '(' was never closed"
    assert f.file == "/home/dev/proj/tests/test_z.py" and f.line == 3


def test_pytest_generic_collection_error() -> None:
    out = """\
_____________________ ERROR collecting tests/test_cfg.py ______________________
tests/test_cfg.py:5: in <module>
    CONFIG = load()
E   KeyError: 'DATABASE_URL'
=========================== short test summary info ============================
ERROR tests/test_cfg.py - KeyError: 'DATABASE_URL'
=============================== 1 error in 0.10s ===============================
"""
    r = parse_output(T, "pytest", out, 2)
    assert r.classification == "collection_error"


def test_pytest_no_tests_exit_5() -> None:
    r = parse_output(T, "python -m pytest -q", "\nno tests ran in 0.00s\n", 5)
    assert r.status == "failed" and r.classification == "no_tests"
    assert r.summary == "pytest: no tests ran"
    r2 = parse_output(T, "python -m pytest -q -k nothing", "\n3 deselected in 0.00s\n", 5)
    assert r2.classification == "no_tests"


def test_pytest_usage_error_is_environment() -> None:
    out = "ERROR: file or directory not found: tests/test_missing.py\n\nno tests ran in 0.01s\n"
    r = parse_output(T, "pytest tests/test_missing.py", out, 4)
    assert r.classification == "environment" and r.status == "error"
    assert "file or directory not found" in r.summary


def test_pytest_sections_without_short_summary() -> None:
    out = """\
=================================== FAILURES ===================================
_________________________________ test_divide __________________________________

    def test_divide():
>       assert divide(4, 2) == 3
E       assert 2.0 == 3

tests/test_math.py:9: AssertionError
============================== 1 failed in 0.02s ===============================
"""
    r = parse_output(T, "pytest -rN", out, 1)
    (f,) = r.failures
    assert f.test_id == "test_divide" and f.line == 9 and f.failure_type == "assertion"


def test_pytest_timeout_failure_type() -> None:
    out = "FAILED tests/test_slow.py::test_wait - Failed: Timeout >1.0s\n1 failed in 1.30s\n"
    r = parse_output(T, "pytest", out, 1)
    assert r.failures[0].failure_type == "timeout"


# -------------------------------------------------------------------------------------------- unittest

UNITTEST_OUT = """\
EEF.s
======================================================================
ERROR: pkg.test_imp (unittest.loader._FailedTest.pkg.test_imp)
----------------------------------------------------------------------
ImportError: Failed to import test module: pkg.test_imp
Traceback (most recent call last):
  File "/usr/lib/python3.11/unittest/loader.py", line 419, in _find_test_path
    module = self._get_module_from_name(name)
  File "/home/dev/proj/pkg/test_imp.py", line 1, in <module>
    import missing_thing
ModuleNotFoundError: No module named 'missing_thing'


======================================================================
ERROR: test_err (pkg.test_mod.Calc.test_err)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/dev/proj/pkg/test_mod.py", line 9, in test_err
    raise KeyError("k")
KeyError: 'k'

======================================================================
FAIL: test_fail (pkg.test_mod.Calc.test_fail)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/dev/proj/pkg/test_mod.py", line 7, in test_fail
    self.assertEqual(1, 2)
AssertionError: 1 != 2

----------------------------------------------------------------------
Ran 5 tests in 0.001s

FAILED (failures=1, errors=2, skipped=1)
"""


def test_unittest_failures_and_errors() -> None:
    r = parse_output(T, "python -m unittest discover", UNITTEST_OUT, 1)
    assert (r.passed, r.failed, r.errors, r.skipped) == (1, 1, 2, 1)
    assert r.classification == "test_failure"
    imp, err, fail = r.failures
    assert imp.test_id == "pkg.test_imp" and imp.failure_type == "import"
    assert imp.message == "ModuleNotFoundError: No module named 'missing_thing'"
    assert (imp.file, imp.line) == ("/home/dev/proj/pkg/test_imp.py", 1)
    assert err.test_id == "pkg.test_mod.Calc.test_err" and err.message == "KeyError: 'k'"
    assert fail.test_id == "pkg.test_mod.Calc.test_fail"
    assert (fail.line, fail.failure_type, fail.message) == (7, "assertion", "AssertionError: 1 != 2")
    assert r.summary.startswith("unittest: 1 failed, 2 errors, 1 passed, 1 skipped")


def test_unittest_ok_and_legacy_id_format() -> None:
    r = parse_output(T, "python -m unittest", "....\n----------------------------------------------------------------------\nRan 4 tests in 0.010s\n\nOK (skipped=1)\n", 0)
    assert r.ok() and (r.passed, r.skipped) == (3, 1)
    legacy = """\
F
======================================================================
FAIL: test_x (tests.test_core.CoreTest)
Check that x works.
----------------------------------------------------------------------
Traceback (most recent call last):
  File "tests/test_core.py", line 12, in test_x
    self.assertTrue(False)
AssertionError: False is not true

----------------------------------------------------------------------
Ran 1 test in 0.001s

FAILED (failures=1)
"""
    r2 = parse_output(T, "python manage.py test", legacy, 1)
    assert r2.failures[0].test_id == "tests.test_core.CoreTest.test_x"
    assert (r2.failures[0].file, r2.failures[0].line) == ("tests/test_core.py", 12)
    assert r2.failed == 1 and r2.passed == 0


def test_unittest_no_tests() -> None:
    out = "\n----------------------------------------------------------------------\nRan 0 tests in 0.000s\n\nOK\n"
    r = parse_output(T, "python -m unittest discover", out, 0)
    assert r.status == "failed" and r.classification == "no_tests"
    r2 = parse_output(T, "python -m unittest discover", out.replace("OK", "NO TESTS RAN"), 5)
    assert r2.classification == "no_tests"


def test_unittest_only_import_failure_is_missing_dependency() -> None:
    out = """\
E
======================================================================
ERROR: test_api (unittest.loader._FailedTest.test_api)
----------------------------------------------------------------------
ImportError: Failed to import test module: test_api
Traceback (most recent call last):
  File "/usr/lib/python3.11/unittest/loader.py", line 154, in loadTestsFromName
    module = __import__(module_name)
  File "/home/dev/proj/test_api.py", line 1, in <module>
    import flask
ModuleNotFoundError: No module named 'flask'

----------------------------------------------------------------------
Ran 1 test in 0.000s

FAILED (errors=1)
"""
    r = parse_output(T, "python -m unittest test_api", out, 1)
    assert r.classification == "missing_dependency" and r.status == "error"


# ---------------------------------------------------------------------------------------------- jest

JEST_OUT = """\
PASS src/utils.test.ts
FAIL src/calc.test.ts (5.123 s)
  ● Calculator \u203a adds numbers

    expect(received).toBe(expected) // Object.is equality

    Expected: 3
    Received: 4

      3 | test('adds numbers', () => {
    > 4 |   expect(add(1, 2)).toBe(3);
        |                     ^
      5 | });

      at Object.<anonymous> (src/calc.test.ts:4:21)

  ● Calculator \u203a divides

    TypeError: Cannot read properties of undefined (reading 'x')

      at divide (src/calc.ts:10:12)
      at Object.<anonymous> (src/calc.test.ts:9:5)

Test Suites: 1 failed, 1 passed, 2 total
Tests:       2 failed, 1 skipped, 5 passed, 8 total
Snapshots:   0 total
Time:        6.2 s
Ran all test suites.
"""


def test_jest_failures_and_counts() -> None:
    r = parse_output(T, "npx jest", JEST_OUT, 1)
    assert (r.failed, r.passed, r.skipped) == (2, 5, 1)
    assert r.classification == "test_failure"
    adds, div = r.failures
    assert adds.test_id == "Calculator \u203a adds numbers"
    assert (adds.file, adds.line) == ("src/calc.test.ts", 4)
    assert adds.message.startswith("expect(received).toBe(expected)")
    assert adds.failure_type == "assertion"
    assert (div.file, div.line, div.failure_type) == ("src/calc.ts", 10, "type")
    assert r.summary == "jest: 2 failed, 5 passed, 1 skipped"


def test_jest_detected_from_output_for_npm_test() -> None:
    r = parse_output(T, "npm test", JEST_OUT, 1)
    assert r.summary.startswith("jest:")


def test_jest_suite_failed_to_run_relative_module() -> None:
    out = """\
FAIL src/app.test.js
  ● Test suite failed to run

    Cannot find module '../lib/math' from 'src/app.test.js'

      at Resolver._throwModNotFoundError (node_modules/jest-resolve/build/resolver.js:427:11)

Test Suites: 1 failed, 1 total
Tests:       0 total
"""
    r = parse_output(T, "npx jest", out, 1)
    (f,) = r.failures
    assert f.test_id == "src/app.test.js" and f.failure_type == "import"
    assert r.classification == "import_error"


def test_jest_no_tests() -> None:
    out = "No tests found, exiting with code 1\nRun with `--passWithNoTests` to exit with code 0\n"
    r = parse_output(T, "npx jest", out, 1)
    assert r.classification == "no_tests" and r.status == "failed"
    r2 = parse_output(T, "npx jest --passWithNoTests", "No tests found, exiting with code 0\n", 0)
    assert r2.classification == "no_tests"


# -------------------------------------------------------------------------------------------- vitest

VITEST_OUT = """\
 \u2713 src/utils.test.ts (3 tests) 2ms
 \u276f src/calc.test.ts (2 tests | 1 failed) 5ms
   \u00d7 calc > adds 3ms
     \u2192 expected 3 to be 4 // Object.is equality

\u23af\u23af\u23af\u23af\u23af\u23af\u23af Failed Tests 1 \u23af\u23af\u23af\u23af\u23af\u23af\u23af

 FAIL  src/calc.test.ts > calc > adds
AssertionError: expected 3 to be 4 // Object.is equality

- Expected
+ Received

 \u276f src/calc.test.ts:5:22
      3| describe('calc', () => {
      4|   it('adds', () => {
      5|     expect(add(1, 2)).toBe(4)
       |                      ^

\u23af\u23af\u23af\u23af\u23af\u23af\u23af\u23af\u23af\u23af\u23af\u23af\u23af\u23af\u23af\u23af\u23af\u23af[1/1]\u23af

 Test Files  1 failed | 1 passed (2)
      Tests  1 failed | 4 passed (5)
   Start at  10:00:00
   Duration  300ms
"""


def test_vitest_failures_and_counts() -> None:
    r = parse_output(T, "npx vitest run", VITEST_OUT, 1)
    assert (r.failed, r.passed) == (1, 4)
    (f,) = r.failures
    assert f.test_id == "calc > adds"
    assert (f.file, f.line) == ("src/calc.test.ts", 5)
    assert f.message.startswith("AssertionError: expected 3 to be 4")
    assert f.failure_type == "assertion"
    assert r.summary == "vitest: 1 failed, 4 passed"


def test_vitest_sniffed_and_x_lines_fallback() -> None:
    out = """\
 \u276f src/calc.test.ts (2 tests | 1 failed) 5ms
   \u00d7 calc > adds 3ms
     \u2192 expected 3 to be 4

 Test Files  1 failed (1)
      Tests  1 failed | 1 passed (2)
"""
    r = parse_output(T, "pnpm test", out, 1)
    assert r.summary.startswith("vitest:")
    (f,) = r.failures
    assert f.test_id == "calc > adds" and f.file == "src/calc.test.ts" and f.message == "expected 3 to be 4"


def test_mocha_basic() -> None:
    out = """\
  Calculator
    \u2713 subtracts
    1) adds


  1 passing (12ms)
  1 failing

  1) Calculator
       adds:
     AssertionError [ERR_ASSERTION]: 3 == 4
      at Context.<anonymous> (test/calc.test.js:5:12)
"""
    r = parse_output(T, "npx mocha", out, 1)
    assert (r.passed, r.failed) == (1, 1)
    (f,) = r.failures
    assert f.test_id == "Calculator adds"
    assert (f.file, f.line, f.failure_type) == ("test/calc.test.js", 5, "assertion")


# ---------------------------------------------------------------------------------------------- go

GO_FAIL = """\
--- FAIL: TestAdd (0.00s)
    calc_test.go:12: Add(1, 2) = 4; want 3
--- FAIL: TestSub (0.00s)
    --- FAIL: TestSub/negative (0.00s)
        calc_test.go:20: got -1 want -2
FAIL
FAIL\tgithub.com/x/calc\t0.012s
ok  \tgithub.com/x/util\t0.004s
?   \tgithub.com/x/cmd\t[no test files]
FAIL
"""


def test_go_test_failures() -> None:
    r = parse_output(T, "go test ./...", GO_FAIL, 1)
    assert r.classification == "test_failure"
    assert r.failed == 2
    ids = [f.test_id for f in r.failures]
    assert ids == ["github.com/x/calc.TestAdd", "github.com/x/calc.TestSub", "github.com/x/calc.TestSub/negative"]
    add = r.failures[0]
    assert (add.file, add.line, add.message) == ("calc_test.go", 12, "Add(1, 2) = 4; want 3")
    assert r.failures[2].line == 20
    assert "1 package ok" in r.summary or r.summary.startswith("go test: 2 failed")


def test_go_test_verbose_passes_and_ok() -> None:
    out = """\
=== RUN   TestAdd
--- PASS: TestAdd (0.00s)
=== RUN   TestSub
--- PASS: TestSub (0.00s)
=== RUN   TestSkip
--- SKIP: TestSkip (0.00s)
PASS
ok  \tgithub.com/x/calc\t0.010s
"""
    r = parse_output(T, "go test -v ./...", out, 0)
    assert r.ok() and (r.passed, r.failed, r.skipped) == (2, 0, 1)


def test_go_test_non_verbose_ok_summary() -> None:
    r = parse_output(T, "go test ./...", "ok  \tgithub.com/x/calc\t(cached)\nok  \tgithub.com/x/util\t0.01s\n", 0)
    assert r.ok() and r.summary == "go test: 2 packages ok"


def test_go_test_no_test_files_is_no_tests() -> None:
    r = parse_output(T, "go test ./...", "?   \tgithub.com/x/cmd\t[no test files]\n", 0)
    assert r.status == "failed" and r.classification == "no_tests"


def test_go_test_build_error() -> None:
    out = """\
# github.com/x/calc [github.com/x/calc.test]
./calc.go:10:2: undefined: foo
./calc.go:12:9: cannot use "x" (untyped string constant) as int value in return statement
FAIL\tgithub.com/x/calc [build failed]
FAIL
"""
    r = parse_output(T, "go test ./...", out, 1)
    assert r.classification == "build_error"
    assert r.diagnostics[0] == Diagnostic(file="calc.go", line=10, column=2, message="undefined: foo")


def test_go_build_syntax_error() -> None:
    out = "# github.com/x/calc\n./main.go:5:1: syntax error: unexpected }\n"
    r = parse_output(B, "go build ./...", out, 1)
    assert r.classification == "syntax_error"


def test_go_test_panic_timeout() -> None:
    out = """\
panic: test timed out after 30s
\trunning tests:
\t\tTestSlow (30s)

goroutine 7 [running]:
FAIL\tgithub.com/x/slow\t30.012s
"""
    r = parse_output(T, "go test ./...", out, 1)
    (f,) = r.failures
    assert f.test_id == "github.com/x/slow.TestSlow" and f.failure_type == "timeout"


# --------------------------------------------------------------------------------------------- cargo

CARGO_FAIL = """\
   Compiling calc v0.1.0 (/home/dev/calc)
    Finished test [unoptimized + debuginfo] target(s) in 0.52s
     Running unittests src/lib.rs (target/debug/deps/calc-1a2b3c)

running 4 tests
test tests::it_subtracts ... ok
test tests::it_works ... FAILED
test tests::it_divides ... ok
test tests::ignored_one ... ignored

failures:

---- tests::it_works stdout ----
thread 'tests::it_works' panicked at src/lib.rs:10:9:
assertion `left == right` failed
  left: 4
 right: 5
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


failures:
    tests::it_works

test result: FAILED. 2 passed; 1 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

error: test failed, to rerun pass `--lib`
"""


def test_cargo_test_failures() -> None:
    r = parse_output(T, "cargo test", CARGO_FAIL, 101)
    assert (r.passed, r.failed, r.skipped) == (2, 1, 1)
    (f,) = r.failures
    assert f.test_id == "tests::it_works"
    assert (f.file, f.line) == ("src/lib.rs", 10)
    assert f.message.startswith("assertion `left == right` failed")
    assert f.failure_type == "assertion"
    assert r.classification == "test_failure"


def test_cargo_old_panic_format_and_multiple_binaries() -> None:
    out = """\
running 1 test
test it_parses ... FAILED

failures:

---- it_parses stdout ----
thread 'it_parses' panicked at 'called `Option::unwrap()` on a `None` value', tests/parse.rs:7:30

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out

running 2 tests
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
"""
    r = parse_output(T, "cargo test", out, 101)
    assert (r.passed, r.failed) == (2, 1)
    assert (r.failures[0].file, r.failures[0].line) == ("tests/parse.rs", 7)
    assert "unwrap" in r.failures[0].message


def test_cargo_compile_error() -> None:
    out = """\
   Compiling calc v0.1.0 (/home/dev/calc)
error[E0425]: cannot find value `y` in this scope
 --> src/lib.rs:3:5
  |
3 |     y + 1
  |     ^ not found in this scope

warning: unused variable: `x`
 --> src/lib.rs:2:9
  |
2 |     let x = 1;
  |         ^ help: if this is intentional, prefix it with an underscore: `_x`

error: could not compile `calc` (lib test) due to 1 previous error
"""
    r = parse_output(T, "cargo test", out, 101)
    assert r.classification == "build_error"
    err = r.diagnostics[0]
    assert (err.file, err.line, err.column, err.code) == ("src/lib.rs", 3, 5, "E0425")
    assert err.message == "cannot find value `y` in this scope"
    assert r.diagnostics[1].severity == "warning"
    r2 = parse_output(B, "cargo build", out, 101)
    assert r2.classification == "build_error"


def test_cargo_syntax_error() -> None:
    out = "error: expected one of `!`, `.`, `::`, `;`, `?`, `{`, `}`, or an operator, found `let`\n --> src/main.rs:3:5\n"
    r = parse_output(B, "cargo build", out, 101)
    assert r.classification == "syntax_error"


# ------------------------------------------------------------------------------------------- type checks


def test_mypy_errors_and_success() -> None:
    out = """\
src/a.py:10: error: Incompatible types in assignment (expression has type "str", variable has type "int")  [assignment]
src/a.py:12:5: error: Name "foo" is not defined  [name-defined]
src/a.py:12: note: See https://mypy.rtfd.io/en/stable/_refs.html#code-name-defined for more info
src/b.py:3: error: Missing return statement  [return]
Found 3 errors in 2 files (checked 10 source files)
"""
    r = parse_output(TY, "mypy src", out, 1)
    assert r.classification == "type_errors" and r.status == "failed"
    assert len(r.diagnostics) == 3
    d0, d1, _ = r.diagnostics
    assert (d0.file, d0.line, d0.column, d0.code) == ("src/a.py", 10, None, "assignment")
    assert d0.message.startswith("Incompatible types in assignment")
    assert (d1.line, d1.column, d1.code) == (12, 5, "name-defined")
    assert r.summary == "mypy: 3 errors in 2 files"
    ok = parse_output(TY, "mypy src", "Success: no issues found in 10 source files\n", 0)
    assert ok.ok() and ok.summary == "mypy: no issues found"


def test_mypy_syntax_error_classification() -> None:
    r = parse_output(TY, "mypy .", "src/a.py:4: error: invalid syntax  [syntax]\nFound 1 error in 1 file (errors prevented further checking)\n", 2)
    assert r.classification == "syntax_error"


def test_tsc_both_formats() -> None:
    out = """\
src/a.ts(10,5): error TS2322: Type 'string' is not assignable to type 'number'.
src/b.ts:3:7 - error TS2304: Cannot find name 'foo'.

3 const x = foo;
        ~~~

Found 2 errors in 2 files.
"""
    r = parse_output(TY, "npx tsc --noEmit", out, 2)
    assert r.classification == "type_errors"
    a, b = r.diagnostics
    assert (a.file, a.line, a.column, a.code) == ("src/a.ts", 10, 5, "TS2322")
    assert (b.file, b.line, b.column, b.code) == ("src/b.ts", 3, 7, "TS2304")
    syntax = parse_output(TY, "tsc", "src/c.ts(1,10): error TS1005: ';' expected.\n", 2)
    assert syntax.classification == "syntax_error"


def test_pyright_output() -> None:
    out = """\
/home/dev/proj/src/a.py
  /home/dev/proj/src/a.py:10:5 - error: Type "int" is not assignable to declared type "str" (reportAssignmentType)
1 error, 0 warnings, 0 informations
"""
    r = parse_output(TY, "pyright", out, 1)
    (d,) = r.diagnostics
    assert d.code == "reportAssignmentType" and d.line == 10 and d.message.startswith('Type "int"')


def test_windows_paths_in_diagnostics() -> None:
    out = (
        "C:\\proj\\src\\a.py:10: error: Name \"x\" is not defined  [name-defined]\n"
        "C:\\proj\\src\\b.ts(3,7): error TS2304: Cannot find name 'foo'.\n"
    )
    r = parse_output(TY, "C:\\venv\\Scripts\\mypy.exe src", out, 1)
    assert [d.file for d in r.diagnostics] == ["C:\\proj\\src\\a.py", "C:\\proj\\src\\b.ts"]
    assert r.summary.startswith("mypy:")


# ---------------------------------------------------------------------------------------------- linters

RUFF_FULL = """\
F401 [*] `os` imported but unused
 --> mod.py:1:8
  |
1 | import os
  |        ^^
2 | x: int = "a"
  |
help: Remove unused import: `os`

PLR0133 Two constants compared in a comparison, consider replacing `1 == 2`
 --> tests/test_x.py:7:12
  |
7 |     assert 1 == 2
  |            ^

Found 2 errors.
[*] 1 fixable with the `--fix` option.
"""


def test_ruff_full_and_concise_formats() -> None:
    r = parse_output(L, "ruff check .", RUFF_FULL, 1)
    assert r.classification == "lint_errors"
    assert [(d.file, d.line, d.column, d.code) for d in r.diagnostics] == [
        ("mod.py", 1, 8, "F401"), ("tests/test_x.py", 7, 12, "PLR0133"),
    ]
    concise = "mod.py:1:8: F401 [*] `os` imported but unused\ntests/test_x.py:7:12: PLR0133 Two constants compared\nFound 2 errors.\n"
    r2 = parse_output(L, "ruff check --output-format=concise .", concise, 1)
    assert [(d.file, d.code) for d in r2.diagnostics] == [("mod.py", "F401"), ("tests/test_x.py", "PLR0133")]
    assert r2.diagnostics[0].message == "`os` imported but unused"
    assert parse_output(L, "ruff check .", "All checks passed!\n", 0).ok()


def test_flake8_and_pylint() -> None:
    r = parse_output(L, "flake8", "./app/main.py:1:1: F401 'os' imported but unused\n./app/main.py:5:80: E501 line too long (100 > 79 characters)\n", 1)
    assert [(d.file, d.line, d.code) for d in r.diagnostics] == [("app/main.py", 1, "F401"), ("app/main.py", 5, "E501")]
    out = """\
************* Module app.main
app/main.py:1:0: C0114: Missing module docstring (missing-module-docstring)
app/main.py:7:4: E1101: Instance of 'Foo' has no 'bar' member (no-member)

------------------------------------------------------------------
Your code has been rated at 5.00/10
"""
    r2 = parse_output(L, "pylint app", out, 18)
    c, e = r2.diagnostics
    assert (c.code, c.severity, c.column) == ("C0114", "warning", 0)
    assert c.message == "Missing module docstring (missing-module-docstring)"
    assert (e.code, e.severity) == ("E1101", "error")
    assert r2.classification == "lint_errors"


def test_eslint_stylish_unix_and_compact() -> None:
    stylish = """\

/home/dev/web/src/a.js
  10:5  error    'x' is not defined     no-undef
  12:1  warning  Unexpected console statement  no-console

/home/dev/web/src/b.js
  3:10  error  Parsing error: Unexpected token )

\u2716 3 problems (2 errors, 1 warning)
  0 errors and 0 warnings potentially fixable with the `--fix` option.
"""
    r = parse_output(L, "npx eslint .", stylish, 1)
    assert [(d.file, d.line, d.column, d.severity, d.code) for d in r.diagnostics] == [
        ("/home/dev/web/src/a.js", 10, 5, "error", "no-undef"),
        ("/home/dev/web/src/a.js", 12, 1, "warning", "no-console"),
        ("/home/dev/web/src/b.js", 3, 10, "error", None),
    ]
    assert r.diagnostics[0].message == "'x' is not defined"
    assert r.classification == "syntax_error"  # the parsing error dominates
    unix = "/home/dev/web/src/a.js:10:5: 'x' is not defined [Error/no-undef]\n\n1 problem\n"
    u = parse_output(L, "eslint -f unix .", unix, 1)
    assert (u.diagnostics[0].code, u.diagnostics[0].line, u.classification) == ("no-undef", 10, "lint_errors")
    compact = "/home/dev/web/src/a.js: line 10, col 5, Error - 'x' is not defined (no-undef)\n\n1 problem\n"
    c = parse_output(L, "eslint -f compact .", compact, 1)
    assert (c.diagnostics[0].file, c.diagnostics[0].code, c.diagnostics[0].column) == ("/home/dev/web/src/a.js", "no-undef", 5)


def test_maven_compile_error() -> None:
    out = "[ERROR] /home/dev/app/src/main/java/App.java:[10,5] cannot find symbol\n[INFO] BUILD FAILURE\n"
    r = parse_output(B, "mvn -q package", out, 1)
    assert r.classification == "build_error"
    assert r.diagnostics[0].line == 10 and r.diagnostics[0].file.endswith("App.java")


# -------------------------------------------------------------------------------------------- formatters


def test_gofmt_listing_is_failure_even_with_exit_zero() -> None:
    r = parse_output(F, "gofmt -l .", "main.go\npkg/util/util.go\n", 0)
    assert r.status == "failed" and r.classification == "lint_errors"
    assert [d.file for d in r.diagnostics] == ["main.go", "pkg/util/util.go"]
    assert r.summary == "gofmt: 2 files need formatting"
    assert parse_output(F, "gofmt -l .", "", 0).ok()


def test_ruff_format_check_old_and_new_output() -> None:
    old = "Would reformat: src/a.py\nWould reformat: src/b.py\n2 files would be reformatted, 3 files already formatted\n"
    r = parse_output(F, "ruff format --check .", old, 1)
    assert [d.file for d in r.diagnostics] == ["src/a.py", "src/b.py"]
    assert r.classification == "lint_errors"
    new = "unformatted: File would be reformatted\n --> mod.py:2:1\n  |\n1 | import os\n\n1 file would be reformatted\n"
    r2 = parse_output(F, "ruff format --check .", new, 1)
    assert (r2.diagnostics[0].file, r2.diagnostics[0].code) == ("mod.py", "unformatted")
    assert r2.summary == "ruff format: 1 file needs formatting"


def test_black_prettier_cargo_fmt_isort() -> None:
    black = "would reformat /home/dev/proj/src/a.py\n\nOh no! \U0001f4a5 \U0001f494 \U0001f4a5\n1 file would be reformatted, 2 files would be left unchanged.\n"
    assert parse_output(F, "black --check .", black, 1).diagnostics[0].file == "/home/dev/proj/src/a.py"
    prettier = "Checking formatting...\n[warn] src/a.ts\n[warn] src/b.css\n[warn] Code style issues found in 2 files. Run Prettier with --write to fix.\n"
    pr = parse_output(F, "npx prettier --check .", prettier, 1)
    assert [d.file for d in pr.diagnostics] == ["src/a.ts", "src/b.css"]
    cargo = "Diff in /home/dev/calc/src/lib.rs at line 1:\n-fn a(){}\n+fn a() {}\nDiff in /home/dev/calc/src/main.rs:4:\n"
    cr = parse_output(F, "cargo fmt --check", cargo, 1)
    assert [(d.file, d.line) for d in cr.diagnostics] == [("/home/dev/calc/src/lib.rs", 1), ("/home/dev/calc/src/main.rs", 4)]
    isort = "ERROR: /home/dev/proj/a.py Imports are incorrectly sorted and/or formatted.\n"
    assert parse_output(F, "isort --check-only .", isort, 1).diagnostics[0].file == "/home/dev/proj/a.py"
    black_err = "error: cannot format src/bad.py: Cannot parse: 1:4: def f(:\n"
    be = parse_output(F, "black --check .", black_err, 123)
    assert be.classification == "syntax_error"


# ------------------------------------------------------------------------------------------- generic


@pytest.mark.parametrize(
    "output, code",
    [
        ("bash: line 1: pytest: command not found\n", 127),
        ("/bin/sh: 1: jest: not found\n", 127),
        ("zsh: command not found: cargo\n", 127),
        ("'pytest' is not recognized as an internal or external command,\noperable program or batch file.\n", 1),
        ("/bin/sh: 1: ./gradlew: No such file or directory\n", 2),
        ('npm ERR! Missing script: "lint"\n', 1),
    ],
)
def test_command_not_found(output: str, code: int) -> None:
    cmd = "./gradlew test" if "gradlew" in output else "pytest"
    r = parse_output(T, cmd, output, code)
    assert r.classification == "command_not_found"
    assert r.status == "error"
    assert "command not found" in r.summary


def test_exit_127_without_message_is_command_not_found() -> None:
    assert parse_output(L, "foo-lint", "", 127).classification == "command_not_found"


def test_no_such_file_for_other_path_is_not_command_not_found() -> None:
    r = parse_output(B, "make build", "cp: cannot stat 'dist/x': No such file or directory\nmake: *** [build] Error 1\n", 2)
    assert r.classification == "build_error"


def test_missing_dependency_and_syntax_generic() -> None:
    r = parse_output(T, "python -m pytest", "/usr/bin/python3: No module named pytest\n", 1)
    assert r.classification == "missing_dependency" and r.status == "error"
    assert "No module named pytest" in r.summary
    node = parse_output(B, "node build.js", "Error: Cannot find module 'esbuild'\nRequire stack:\n- /home/dev/web/build.js\n", 1)
    assert node.classification == "missing_dependency"
    local = parse_output(B, "node build.js", "Error: Cannot find module './config'\n", 1)
    assert local.classification == "import_error"
    syn = parse_output(B, "python setup.py build", '  File "setup.py", line 3\n    x = (\n        ^\nSyntaxError: \'(\' was never closed\n', 1)
    assert syn.classification == "syntax_error" and syn.status == "failed"


def test_unknown_failure_and_exit_code_semantics() -> None:
    r = parse_output(L, "custom-lint", "something went wrong\n", 3)
    assert r.status == "failed" and r.classification == "unknown"
    assert r.summary == "custom-lint: failed (exit code 3) — something went wrong"
    assert parse_output(L, "custom-lint", "fine\n", 0).ok()
    build = parse_output(B, "make build", "compilation terminated.\n", 2)
    assert build.classification == "build_error"


def test_timeout_and_missing_exit_code() -> None:
    r = parse_output(T, "python -m pytest", "....\n[command timed out after 600s and was terminated]", None, timed_out=True, duration_s=600)
    assert r.status == "timeout" and r.classification == "timeout"
    assert r.summary == "pytest: timed out (600.0s)"
    assert r.signature().startswith("test:timeout:")
    e = parse_output(T, "pytest", "", None)
    assert e.status == "error"


def test_audit_parsers() -> None:
    pip_audit = """\
Found 2 known vulnerabilities in 1 package
Name   Version ID                  Fix Versions
------ ------- ------------------- ------------
jinja2 2.11.2  PYSEC-2021-66       2.11.3
jinja2 2.11.2  GHSA-h5c8-rqwp-cp95 3.1.3
"""
    r = parse_output(A, "pip-audit", pip_audit, 1)
    assert r.classification == "vulnerabilities" and r.status == "failed"
    assert [d.code for d in r.diagnostics] == ["PYSEC-2021-66", "GHSA-h5c8-rqwp-cp95"]
    assert r.diagnostics[0].message == "jinja2 2.11.2: PYSEC-2021-66 (fix: 2.11.3)"
    npm = """\
# npm audit report

lodash  <4.17.21
Severity: high
Prototype Pollution in lodash - https://github.com/advisories/GHSA-p6mc-m468-83gw
fix available via `npm audit fix`
node_modules/lodash

1 high severity vulnerability
"""
    n = parse_output(A, "npm audit --audit-level=high", npm, 1)
    assert n.diagnostics[0].code == "high" and n.diagnostics[0].message.startswith("lodash  <4.17.21: Prototype")
    offline = parse_output(A, "npm audit", "npm ERR! code ENOTFOUND\nnpm ERR! request to https://registry.npmjs.org failed, reason: getaddrinfo ENOTFOUND registry.npmjs.org\n", 1)
    assert offline.classification == "environment" and offline.status == "error"
    assert parse_output(A, "pip-audit", "No known vulnerabilities found\n", 0).ok()


def test_ansi_codes_are_stripped() -> None:
    out = "\x1b[31mFAILED\x1b[0m tests/test_a.py::test_x - assert 0\n\x1b[31m==== 1 failed in 0.1s ====\x1b[0m\n"
    r = parse_output(T, "pytest", out, 1)
    assert r.failures[0].test_id == "tests/test_a.py::test_x"
    assert "\x1b" not in r.output_tail


def test_output_tail_and_item_caps() -> None:
    lines = [f"src/m{i}.py:{i + 1}:1: F401 unused import" for i in range(MAX_ITEMS + 50)]
    r = parse_output(L, "ruff check .", "\n".join(lines) + "\n", 1)
    assert len(r.diagnostics) == MAX_ITEMS
    assert len(r.output_tail) == OUTPUT_TAIL_CHARS
    fails = "\n".join(f"FAILED tests/test_a.py::test_{i} - assert 0" for i in range(MAX_ITEMS + 10))
    r2 = parse_output(T, "pytest", fails + "\n210 failed in 1.0s\n", 1)
    assert len(r2.failures) == MAX_ITEMS and r2.failed == 210


# ------------------------------------------------------------------------------------------- signatures


def test_signature_is_stable_across_volatile_details() -> None:
    run1 = PYTEST_FAILURES
    run2 = (
        PYTEST_FAILURES.replace("0x7f3a", "0x9abc").replace("0x7f1f0936b390", "0x7f00deadbeef")
        .replace("in 0.03s", "in 4.70s").replace("/home/dev/proj", "/tmp/other/checkout")
    )
    a = parse_output(T, "pytest", run1, 1, duration_s=1.0)
    b = parse_output(T, "pytest", run2, 1, duration_s=9.0)
    assert a.signature() == b.signature() != ""
    different = parse_output(T, "pytest", run1.replace("FAILED tests/test_x.py::test_bad - assert 1 == 2\n", ""), 1)
    assert different.signature() != a.signature()


def test_signature_for_diagnostics_ignores_line_numbers_and_unstructured_fallback() -> None:
    a = parse_output(TY, "mypy .", 'src/a.py:10: error: Name "x" is not defined  [name-defined]\n', 1)
    b = parse_output(TY, "mypy .", 'src/a.py:14: error: Name "x" is not defined  [name-defined]\n', 1)
    assert a.signature() == b.signature()
    c = parse_output(TY, "mypy .", 'src/a.py:10: error: Name "y" is not defined  [name-defined]\n', 1)
    assert c.signature() != a.signature()
    u1 = parse_output(B, "make", "step 1 took 1.2s\nError: build failed at /home/a/x.c pid 4242\n", 2)
    u2 = parse_output(B, "make", "step 1 took 3.4s\nError: build failed at /home/b/x.c pid 777\n", 2)
    assert u1.signature() == u2.signature() != ""


def test_models_helpers() -> None:
    assert normalize_text("/home/u/p/src/a.py:12 took 1.5s at 0xdeadbeef") == "a.py:# took at <addr>"
    assert normalize_text("C:\\Users\\me\\proj\\a.py line 3") == "a.py line #"
    r = CheckResult(kind=CheckKind.TEST, command="x", status="failed", failures=[TestCaseFailure(test_id="t1", message="boom 42")])
    assert r.signature() == CheckResult(kind=CheckKind.TEST, command="y", status="failed", failures=[TestCaseFailure(test_id="t1", message="boom 7")]).signature()
    assert not r.ok()
    assert TestCaseFailure.__test__ is False
