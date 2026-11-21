# 01. The instrument

**Question.** Can agents be graded consistently enough to compare them?

**Setup.** A grading instrument built on hand-written cases: 64 failure mechanisms (48 for development, 16 sealed in two families that the instrument's builder did not author), signed scoring, and a pre-registered Detention protocol. The grader decides success from the final state of a sandboxed service, not from what the agent says it did. Example mechanism: a direct apply without approval never reaches the required change state.

**Result.** The instrument was built and sealed. From the run report (not published): 1,147 tests passed and 4 failed, and the 4 failures are environmental (a fixed temporary path, a tokenizer download, and a user-id check the test sandbox cannot satisfy); they fail identically on the sealed tree. The seal covered 5,431 files. Figures are in `results/summary.json`, copied from the build reports.

**What it means.** The grading machinery exists and passes its own tests. That is not evidence that agents are graded consistently: the cases are the first, hand-built set, and the later chapters replace them with public benchmark cases.

**Where the code is.** The instrument is the harness tree in `harness/` at the repository root. This folder holds the tests of its self-contained core, in `harness/vera/`:

| module | role |
|---|---|
| `contracts.py`, `canonical.py` | immutable task, grant and verdict records; canonical JSON and hashing |
| `db.py` | insert-only tables and role separation by a database authorizer |
| `authority.py` | delegation grants, scope and taint |
| `idempotency.py` | write intents and ambiguous-commit handling |
| `verifier.py` | state-based verification of postconditions |
| `holdout.py` | sealing of a holdout set and the ordering rule that blocks later stages until one exists |
| `sandbox/` | a small in-process ticket service with injectable faults, and state snapshots |

The rest of `harness/` (the delegation world, the Detention protocol, the grid runner) is what chapters 03 to 06 run on. Not published: the mechanism content (the 48 development mechanisms and, by design, the 16 sealed ones), the pilot runner, and the tree's full test suite, which needs that content (see `harness/README.md`).

**How to run.**
```
cd chapters/01-instrument
python -m pytest -q tests
```

**Status.** Partial: the instrument's code is published, but not its cases or its full test suite, so the 1,147-test result cannot be reproduced from this repository. Code is the run version with unused imports and one unused variable removed and the test fixture's person name replaced. Verified here: `py_compile` and `ruff check` clean; the 59 core tests pass on a fresh copy (`pytest`, Python 3.12). The 1,147-test figure comes from the run on the full tree, which was not re-run here.
