# The harness tree (VERA)

This is the code the Agent Detention runs executed: the delegation world, the Detention protocol, the grid runner and the instrument core. Chapters 03 to 06 import it; chapter 01 tests its core. The suite in `src/agent_detention/` uses it as a library and does not modify it.

It is the part of the full tree that those chapters need, taken from the tree the runs used (the sealed tree plus the four-file change for the public-benchmark suite, described below). It was selected by tracing every file the chapter 03 to 06 commands and tests open, then adding the modules those files import on paths the trace did not reach (the live-run preflight, the post-episode auditor, the instrument core).

## Layout

| path | role |
|---|---|
| `vera/contracts.py`, `canonical.py`, `db.py`, `authority.py`, `idempotency.py`, `verifier.py`, `holdout.py`, `errors.py`, `sandbox/` | the instrument core (chapter 01): records, hashing, insert-only storage, grants, state-based verification, holdout sealing |
| `vera/mechanisms/` | state-transition graphs, readiness and solvability checks for a case |
| `vera/pilot/` | instance loading (with the per-variant overrides the suite needs), episode runtime, action parsing, the NVIDIA-hosted route |
| `vera/delegation/world.py`, `policies.py`, `plan.py`, `conditions.py`, ... | the delegation world an episode runs in, and the task conditions |
| `vera/delegation/detention.py`, `heldback.py` | the Detention protocol (review, retry, sibling transfer) and the post-episode auditor |
| `vera/delegation/runner_v13.py`, `runplan_v13.py`, `preregistration.py` | the grid runner, run plans and pre-registration records |
| `vera/delegation/adapters.py`, `routes/` | model adapters (mock, NVIDIA-hosted, Anthropic API), the route registry and the route preflight |
| `tools/delegation/floors.py` | scripted control policies; chapter 04 reuses them |

About 32,000 lines of Python. The only dependency outside the standard library is numpy.

## Changes from the run version

- **Per-variant overrides** (`vera/pilot/instances.py`, `vera/delegation/{case_text,adapters,runner_v13}.py`): the four files changed from the sealed tree so that the two variants of a public-benchmark pair can carry their own entry tables, goal text and forbidden actions. This change is as it was in the run; the edits below touch two of these files for other reasons.
- **No home paths or user names.** The route registry and one comment named the author's home folder or user account; both were removed.
- **Model routes.** The adapter class the live runs used to reach the models, and the route settings and preflight checks that belonged to it, are not published. In their place `adapters.py` has `AnthropicApiAdapter`: one Messages API request per model turn, the harness's own system prompt, no tools, and the key read from the environment variable the route names (`ANTHROPIC_API_KEY`). The pilot registry (`routes/registry_claude_only.json`) keeps the run's route ids and model ids and points both routes at it. `routes/preflight.py` now checks each route's declared transport and adapter and, with `--probe`, makes one short call per route and records the model the provider reports. The run-plan and pre-registration builders name this the route protocol. The gated paid-adapter interface that was also in `adapters.py` (`PaidAdapter`, `PaidBlocked`, `PaidProviderNotWired`, `load_paid_approval` and its approval-file kind) was removed at the same time; no published code used it. The default transport follows no redirect, so the key is sent only to the Messages API address, and error text is cleaned of the key before it is recorded. None of this changes the case files, the suite manifests or the Stage 0 rows, which still rebuild byte-identical. The published numbers come from the runs and were not re-run with this adapter.
- **Development data not published.** Two values the runs read from the development mechanism sets are recorded in code instead: the Detention pre-registration's list of three set-12 mechanisms excluded as two-fix cases (`detention.py`, `SET12_TWO_FIX_MECHANISMS`; recomputed when the set file is present), and the development source-set identity the suite's provenance block carries (`src/agent_detention/suite/integrate/build_inputs.py`). With both, the suite manifests and the Stage 0 row files rebuild byte-identical to the run's.
- **Lint.** Unused imports and eight unused local variables removed (calls with side effects kept), import order normalised, and one dictionary key that was always overwritten by a later one removed. The instrument core is the copy that chapter 01 cleaned (unused imports, one unused variable).
- **Comments.** The decorative `# -----` banner lines were deleted (96 lines in 14 files; the parsed code is unchanged), and three sentences in `adapters.py` that described the build workflow rather than the code were reworded.
- Nothing else was refactored. The code keeps its other run-era comments, which refer to build jobs by number (`JOB25`, `M11`) and to the design documents, which are not published.

## Not published

- The mechanism content: the 48 development mechanisms (`manifests/reliability/`) and, by design, the 16 sealed ones. The instrument's own hand-built cases are therefore not runnable from this repository; the public-benchmark suite is.
- The run artefacts, checkpoints, logs and per-episode records (`artifacts/`, 183 MB), and the design and review documents.
- The pilot runner for the small-model pilots of chapter 02, including its proxy client, and `vera/pilot/agents.py`. `tools/delegation/floors.py` imports the latter only in `evaluate()`, which computes the floors over the development sets and is not used here.
- The owner-review tooling that admits development sets 11 and 12 (`vera/mechanisms/approval.py`, `readiness_v3.py`, `tools/review/`). `vera/delegation/instrument.py` calls it only when a review marker file exists; none is published, so those checks return false and the suite does not reach them.
- The rest of the tree's test suite (1,147 tests passed in the run, 4 environmental failures). Its tests need the development manifests and the run artefacts. The six core test files are in `chapters/01-instrument/tests` (59 tests); the suite tests in `tests/` exercise the rest of this folder end to end.
- The WSP-D environments, release tooling and research tools, which no chapter here uses.

## Running

Nothing needs to be installed for the tree itself beyond numpy (see `requirements.txt`). The suite finds it at `harness/` next to `src/` (a source checkout or an editable install); set `AD_TREE` to use another copy. `python -m agent_detention.suite.integrate.build_inputs` writes the suite's three input manifests into `harness/manifests/suite/`, which is git-ignored.

A live run (`run_suite --pilot`) needs `ANTHROPIC_API_KEY` in the environment, a runner config, a signed pilot gate and a probed preflight report. The report comes from `PYTHONPATH=harness python -m vera.delegation.routes.preflight --profile claude_only --probe --require-measured --out work/preflight.json`, which makes one short call per route. The pilot gate is a JSON file the owner signs; `_validate_pilot_gate_mapping` in `runner_v13.py` lists its fields. The gates and configs of the published runs are not published. Without all four the runner refuses to start a live run. `AnthropicApiAdapter` does not compute cost (its usage records say paid with a cost of 0), and no gate in the tree limits spending: the pilot gate's `model_calls_approved` only allows calls. Cap them with `max_model_calls` in the runner config. The dry run (`--dry-run`) uses mock adapters and calls no model.

## Licence

MIT, as the rest of this repository. All code here was written for this project; it vendors no third-party code.
