# Demonstration

`aie demo` runs the whole pipeline end to end on a small project, offline and
deterministically:

```bash
aie demo                      # creates the project in a new temporary directory
aie demo --dir ./shop -v      # or in a directory of your choice, with more detail
```

**What is real and what is scripted.** The model is a fixed scripted transcript (the
`scripted` provider), so the demo needs no API key or network and always behaves the same.
Everything else is real:
- repository discovery;
- file edits through the guarded tools;
- the project's `unittest` runs;
- failure records;
- checkpoints and commits on a task branch;
- quality gates;
- the report.

The demo shows what the harness does with a model's actions. It does not measure how good
any real model is; for that, run `aie bench run --suite model` with a configured provider.

## The task

> Stock can currently go negative: make Inventory.remove() raise ValueError when more units are removed than are in stock (leaving the stock unchanged), and add Inventory.total_units() returning the number of units across all SKUs, with a test.

The project's `Inventory.remove()` lets stock go negative (one of its three tests fails
before any change), and `total_units()` does not exist yet.

## What the run demonstrates

| Step | Where to see it below |
|---|---|
| 1. Repository discovery | "Inspecting the repository", baseline test run (1 failed, 2 passed) |
| 2. Planning | "Plan with 2 subtask(s)" |
| 3. Implementation | "Implementing s1" / "Implementing s2", file edits |
| 4. Testing | targeted then full `unittest` runs after each subtask |
| 5. Intentional failure | the scripted first fix uses `>=` instead of `>`: "1 error, 2 passed" |
| 6. Diagnosis | "Diagnosing", the debugger reads the code and re-runs the failing test |
| 7. Repair | "Fix attempt 1: root cause: the guard used >= …" |
| 8. Re-test | "3 passed", then "4 passed" after s2 |
| 9. Review | an independent review per subtask and a final review against the requirements |
| 10. Checkpoint | checkpoints before/after each subtask, one verified commit per subtask |
| 11. Final verification | "Final QA": full test run, security scan, final review, quality gates |
| 12. Report generation | the Markdown and JSON engineering report |

## Captured output

The output below is from an actual run. Paths are shortened to `<demo-dir>` and the
interpreter path to `python`.

```text
Demo project: <demo-dir>/shop
Model: a fixed scripted transcript (offline, deterministic); tools, tests, git and gates are real.

  ▶ Starting: Prevent negative stock and add total_units()
  ● Understanding the request
  ✎ change · medium: Prevent negative stock in Inventory.remove() and add Inventory.total_units() with a test
  ● Inspecting the repository and environment
    · Repository: 6 files; primary language python; frameworks: none detected
    · Running baseline checks before any change
    ⧗ Running baseline tests
  ✘ baseline unittest: 1 failed, 2 passed (0.1s)
  ● Planning
  ☰ Plan with 2 subtask(s) (model)
  ● Executing the plan
    · Working on new branch aie/prevent-negative-stock-and-add-total-uni-yqses6 (from main)
    ⚑ checkpoint xgfs1bdn: task start
  ◆ Subtask s1: Reject removing more stock than available
    ⚑ checkpoint bds70nap: before s1: Reject removing more stock than available
  ● Implementing s1
    → Reading inventory/stock.py
    → Editing inventory/stock.py
    ✏ edited inventory/stock.py
    ⧗ Running targeted tests
  ✘ unittest: 1 error, 2 passed (0.1s)
  ⚠ Diagnosing: unittest: 1 error, 2 passed (0.1s)
    → Reading inventory/stock.py
    → Running tests: python -m unittest -v tests.test_stock
    → Editing inventory/stock.py
    ✏ edited inventory/stock.py
    → Running tests: python -m unittest -v tests.test_stock
  🔧 Fix attempt 1: root cause: the guard used >=, so removing exactly the available stock was rejected; it now uses >
    ⧗ Running targeted tests
  ✔ unittest: 3 passed (0.1s)
    ⧗ Running tests
  ✔ unittest: 3 passed (0.1s)
    · Fix attempt 1 resolved the failure
  🔍 Independent review of s1 (round 1)
  🔍 Review: approve (0 blocking, 1 total)
    ⚑ checkpoint e463xnhm: after s1 (COMPLETED)
  ⎇ Committed s1 as 0400107fae
  ◆ Subtask s1 COMPLETED
  ◆ Subtask s2: Add total_units() with a test
    ⚑ checkpoint hfcbk3qt: before s2: Add total_units() with a test
  ● Implementing s2
    → Reading inventory/stock.py
    → Editing inventory/stock.py
    ✏ edited inventory/stock.py
    → Reading tests/test_stock.py
    → Editing tests/test_stock.py
    ✏ edited tests/test_stock.py
    → Running tests: python -m unittest discover -s tests -t . -v
    ⧗ Running targeted tests
  ✔ unittest: 4 passed (0.1s)
    ⧗ Running tests
  ✔ unittest: 4 passed (0.1s)
  🔍 Independent review of s2 (round 1)
  🔍 Review: approve (0 blocking, 0 total)
    ⚑ checkpoint rwb17t33: after s2 (COMPLETED)
  ⎇ Committed s2 as 2bef0b3754
  ◆ Subtask s2 COMPLETED
  ● Final QA: full validation, security scan, final review
    ⧗ Running tests
  ✔ unittest: 4 passed (0.0s)
  🔍 Final review against the original requirements
  🔍 Final review: approve
  ☑ Quality gates: COMPLETED
  📄 Report written to .agent/reports/task_1m41mrda09hyqses6.md
  ■ Task COMPLETED

Task task_1m41mrda09hyqses6: COMPLETED
  PASSED       requirements    requirements and acceptance criteria recorded
  PASSED       implementation  2/2 subtask(s) completed; 2 file(s) changed
  SKIPPED      typecheck       no typecheck command detected
  SKIPPED      lint            no lint command detected
  PASSED       tests           unittest: 4 passed (0.0s)
  SKIPPED      build           no build command detected
  PASSED       security        no high-severity findings in changed code
  PASSED       review          approved (model+deterministic)
  SKIPPED      docs            no documentation impact identified
  PASSED       git_state       branch aie/prevent-negative-stock-and-add-total-uni-yqses6; agent changes committed on aie
  subtask s1: completed; repair iterations 1; commit 0400107fae
  subtask s2: completed; repair iterations 0; commit 2bef0b3754
Report: <demo-dir>/shop/.agent/reports/task_1m41mrda09hyqses6.md
Explore it: cd <demo-dir>/shop && aie logs && aie checkpoint list && git log --oneline
```

Commits on the task branch (`git log --oneline` in the demo project):

```text
2bef0b3 aie: Add total_units() with a test
0400107 aie: Reject removing more stock than available
6fe8911 Demo shop
```

## The generated report

## Engineering report: Prevent negative stock and add total_units()

- **Task ID:** `task_1m41mrda09hyqses6`
- **Status:** **COMPLETED**
- **Created:** 2026-10-03T19:44:48+00:00  **Report written:** 2026-10-03T19:44:50+00:00
- **Duration:** 1.8s
- **Models (role → fallback chain):** default: demo:transcript; planner: demo:transcript; coder: demo:transcript; debugger: demo:transcript; reviewer: demo:transcript; fast: demo:transcript; classifier: demo:transcript; summarizer: demo:transcript; embeddings: demo:transcript
- **Branch:** `aie/prevent-negative-stock-and-add-total-uni-yqses6` (from `main`)

### Summary
Prevent negative stock in Inventory.remove() and add Inventory.total_units() with a test

All configured quality gates ran and passed.

### Requirements
- remove() rejects removing more than available
- total_units() sums all stock

### Acceptance criteria
- remove() raises ValueError and leaves stock unchanged when removing more than available — **met** (inventory/stock.py; tests/test_stock.py pass)
- removing exactly the available stock works — **met** (inventory/stock.py; tests/test_stock.py pass)
- total_units() returns the sum of all stock and is tested — **met** (inventory/stock.py; tests/test_stock.py pass)

### Plan and subtasks
Approach: Guard remove() before mutating state; add total_units() as a read-only sum; cover both with unittest.  (plan source: model)

| Subtask | Status | Repairs | Reviews | Files | Commit |
|---|---|---|---|---|---|
| s1: Reject removing more stock than available | completed | 1 | 1 | 1 | 0400107fae |
| s2: Add total_units() with a test | completed | 0 | 1 | 2 | 2bef0b3754 |

**s1 — agent summary (claims, independently verified below):** remove() now refuses to go below zero
**s2 — agent summary (claims, independently verified below):** added total_units() and test_total_units

### Changes
```
inventory/stock.py  | 5 +++++
 tests/test_stock.py | 7 +++++++
 2 files changed, 12 insertions(+)
```

### Baseline (before any change)
- **test**: failed (2 passed, 1 failed, 0 errors, 0 skipped) — `python -m unittest discover -s tests -t . -v` — unittest: 1 failed, 2 passed (0.1s)
- **lint**: unavailable — no lint command detected
- **typecheck**: unavailable — no typecheck command detected

### Final validation (commands actually executed)
- **test**: passed (4 passed, 0 failed, 0 errors, 0 skipped) — `python -m unittest discover -s tests -t . -v` — unittest: 4 passed (0.0s)
- **lint**: unavailable — no lint command detected
- **typecheck**: unavailable — no typecheck command detected
- **build**: unavailable — no build command detected

### Quality gates
| Gate | Mode | Status | Detail |
|---|---|---|---|
| requirements | required | **PASSED** | requirements and acceptance criteria recorded |
| implementation | required | **PASSED** | 2/2 subtask(s) completed; 2 file(s) changed |
| typecheck | if_available | **SKIPPED** | no typecheck command detected |
| lint | if_available | **SKIPPED** | no lint command detected |
| tests | required | **PASSED** | unittest: 4 passed (0.0s) |
| build | if_available | **SKIPPED** | no build command detected |
| security | required | **PASSED** | no high-severity findings in changed code |
| review | required | **PASSED** | approved (model+deterministic) |
| docs | if_available | **SKIPPED** | no documentation impact identified |
| git_state | required | **PASSED** | branch aie/prevent-negative-stock-and-add-total-uni-yqses6; agent changes committed on aie/prevent-negative-stock-and-add-total-uni-yqses6 |

### Review
Verdict: **approve** (model+deterministic, demo:transcript) — Both requirements are met and covered by tests

### Failures and fixes
#### Attempt 1 (s1) — fixed
- Error: unittest: 1 error, 2 passed (0.1s)
- Likely causes: the implementation does not satisfy what the failing test asserts; an edge case (empty input, None, boundary values) is not handled; a recent change broke behaviour that other code relies on
- Hypothesis: root cause: the guard used >=, so removing exactly the available stock was rejected; it now uses >
- Change: python -m unittest: all tests pass

### Unverified items and limitations
- none

### Checkpoints (rollback)
Restore any checkpoint with `aie checkpoints restore <id>` (a safety checkpoint is taken first).

- `ckpt_1m41mrde0xgfs1bdn` — task start
- `ckpt_1m41mrderbds70nap` — before s1: Reject removing more stock than available
- `ckpt_1m41mre61e463xnhm` — after s1 (COMPLETED) (verified)
- `ckpt_1m41mrecahfcbk3qt` — before s2: Add total_units() with a test
- `ckpt_1m41mres1rwb17t33` — after s2 (COMPLETED) (verified)

### Metrics
- Model calls: 19; tokens: 79464 in / 1149 out (partly estimated); model latency: 0.0s
- Retries: 0; fallbacks: 0
- Tool calls: 11 (errors 1, denied 0)
- Test runs: 7; fix attempts: 1; loops detected: 0
- Context resets: 0; human interventions: 0
