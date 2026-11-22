# Agent Detention

**Question.** When has an AI agent earned the right to act alone? Each agent faces the same cases as its peers and must decide whether to act or hold; when it fails it writes a root-cause analysis before it may retry; a held-back check, not its own grader, decides whether it improved. The open research question is whether a structured root-cause review helps an agent recover better than a plain retry or free-form reflection, and whether the agent reports honestly what it actually did.

**Finding so far.** The Detention step itself has not been tested: the first public-benchmark act-or-hold cases proved too thin for it, and it moves to larger multi-step benchmarks. What has been measured is the grading around it. On the same public act-or-hold cases, Claude Sonnet 5.5 scored 34 of 42 episodes correct (81%) and Claude Haiku 5.5 22 of 41 (54%), strictly graded, in the first block of a pre-registered comparison (83 of 380 planned episodes; the second block stopped on a usage limit). Haiku stopped when it should have acted on half of the act versions (10 of 20), against 2 of 21 for Sonnet. Neither misreported: 0 of 83 reports disagreed with the end state (95% upper bound 4.3%). The earlier work found and fixed two process failures that matter for the numbers: an answer leak in the case text, and a leak audit that checked nothing (see the changelog).

## Chapters

| # | chapter | question | finding | status |
|---|---|---|---|---|
| 01 | [instrument](chapters/01-instrument) | Can agents be graded consistently enough to compare them? | Built and sealed: 64 failure mechanisms, 1,147 tests passed (4 environmental failures). Code published; mechanism content not | partial |
| 02 | [small-model-pilots](chapters/02-small-model-pilots) | Does the instrument run end to end on real models, and do the controls score where they must? | Two 432-episode pilots; models scored below a heuristic because of a prompt/validator mismatch | provisional |
| 03 | [public-benchmark-suite](chapters/03-public-benchmark-suite) | Can public benchmarks become act-or-hold pairs the instrument can grade? | 264 pairs; 95 of 131 Tier 1 pairs usable after a solvability scan | built |
| 04 | [offline-control-gate](chapters/04-offline-control-gate) | Does the grader score obvious policies correctly before any model is tested? | Always-act 100/0, always-hold 0/100, no-op 0/0, coin 48.3% | passed |
| 05 | [live-smoke-run](chapters/05-live-smoke-run) | Does the full pipeline run end to end with a live model? | Clean re-run: 41 calls, full Detention cycle, audit clean on every call | clean re-run passed |
| 06 | [stage0-side-by-side](chapters/06-stage0-side-by-side) | How do two models compare on the same act-or-hold cases? | Sonnet 34/42 vs Haiku 22/41 strict; 0 misreports in 83 | partial: 83 of 380 episodes |

Not in this repository: a first anecdote about failures in multi-agent workflows, an independent review of the instrument (its outcome is in the changelog), and the Detention experiment itself, which has not been run.

## Results

| measure | value | chapter |
|---|---|---|
| failure mechanisms (development / sealed) | 64 (48 / 16) | 01 |
| pilot passes of 72 (heuristic / random / gpt-oss-20b / Nemotron / always-act / abstain-first) | 36 / 29 / 21 / 16 / 9 / 0 | 02 |
| Tier 1 pairs adapted / excluded as read-only / usable after scan | 131 / 132 / 95 of 131 | 03 |
| offline controls, always-act / always-hold / no-op / coin | 100/0, 0/100, 0/0, 48.3% | 04 |
| smoke re-run calls (main / Detention) | 22 / 19 | 05 |
| Stage 0 strict accuracy, Sonnet 5.5 / Haiku 5.5 | 34/42 (81%) / 22/41 (54%) | 06 |
| Stage 0 misreports | 0 of 83 (upper bound 4.3%) | 06 |

Every number traces to the `results/` folder of its chapter or to a build report named there. Nothing here is leaderboard-comparable: cases are projections of public benchmarks into act-or-hold pairs, solo agents, no real side effects. The cases may be in the models' training data, and Stage 0 is one pass per episode.

## Layout

```
harness/                  the harness tree the runs executed (VERA): world, Detention protocol, grid runner, instrument core
src/agent_detention/      the case suite (Tier 1 AgentAbstain, Tier 2 tau2-bench and CRM), audit, row generator
chapters/NN-slug/         one folder per experiment: README, entry scripts, results/
tests/                    suite tests (synthetic fixture; some need the datasets) and offline route tests
scripts/fetch_data.py     downloads the public datasets at the commits the results used
```

## The harness tree

`harness/` holds the part of the harness tree that chapters 01 and 03 to 06 use, about 32,000 lines, cleaned for publication. The suite finds it there; `AD_TREE` overrides the location. What was selected, what changed from the run version and what is left out (the instrument's own mechanism content, the run artefacts, the pilot runner) is in `harness/README.md`.

## Model access

The experiments called each model through an API adapter: one single-turn request per model turn, the harness's own system prompt, no tools. The adapter class the runs used is not published; in its place this repository provides `AnthropicApiAdapter` (`harness/vera/delegation/adapters.py`), which calls the Anthropic Messages API and reads the key from the `ANTHROPIC_API_KEY` environment variable and from nowhere else. The mock adapters run everything offline. The published numbers come from the runs; none was re-run with this adapter.

## How to run

From a fresh clone, with Python 3.12. No step calls a model, and nothing needs a network after the download.

```
pip install -r requirements.txt
pip install -e .
python -m pytest -q chapters/01-instrument/tests       # 59 tests, no data needed
python scripts/fetch_data.py --dest data
export AGENTABSTAIN_DATA=data/agentabstain-data AGENTABSTAIN_REPO=data/agentabstain
export TAU2_ROOT=data/tau2-bench CRM_BENCH_ROOT=data/crm-agent-bench AD_WORK=work
# chapter 03: build the cases
python chapters/03-public-benchmark-suite/build_tier1_records.py --out work/tier1/tier1_set.json
python -m agent_detention.suite.tier2.build_tier2
python chapters/03-public-benchmark-suite/scan_solvability.py --out work/solvability.json --table work/solvability_table.md
# chapter 04: the offline control gate
python chapters/04-offline-control-gate/compat_test.py --refs eligible20 --out work/gate.md
# chapter 05: the full pipeline as a dry run with mock adapters, then the leak audit
python -m agent_detention.suite.integrate.build_inputs
python -m agent_detention.suite.integrate.select_tier1_smoke
python -m agent_detention.suite.integrate.rows --ids-file work/reports/tier1_smoke10/ids.json --slot O1 --repeats 1 --out work/reports/rows.json
python -m agent_detention.suite.integrate.run_suite --assignment work/reports/rows.json --out work/rundry --dry-run --detention
python -m agent_detention.suite.integrate.audit_run work/rundry --json work/audit.json
# chapter 06: the Stage 0 rows, and scoring (here on the dry run)
python chapters/06-stage0-side-by-side/build_rows.py --sample chapters/06-stage0-side-by-side/results/stage0_sample.json --out-dir work/stage0
python chapters/06-stage0-side-by-side/score.py --run B1=work/rundry --rows B1=work/reports/rows.json --out work/scores.json
# tests, once the files above exist
python -m pytest -q tests chapters/03-public-benchmark-suite/test_scan_solvability.py
```

The chapter READMEs give each step's options and what it reproduces. The Stage 0 row files rebuild byte-identical to the ones the run used. `python -m agent_detention.suite.integrate.build_inputs --check` compares the suite inputs in `harness/manifests/suite/` with a rebuild and leaves the folder as it found it. CI (`.github/workflows/ci.yml`) runs ruff and `pytest` with no data; the data-dependent tests skip. `tests/routes/test_live.py` makes one real, paid call to the Messages API and runs only when both `ANTHROPIC_API_KEY` and `AD_LIVE_SMOKE=1` are set. A live run needs `ANTHROPIC_API_KEY`, a runner config, a signed pilot gate and a probed preflight report; the gates and configs of the published runs are not in this repository (see `harness/README.md`).

## Licence and dataset credits

Code: MIT, Copyright (c) 2026 Mohammad Walid. This includes `harness/`, which was written for this project and vendors no third-party code.

The licences below were read from the licence files in the local clones the results used; they have not yet been checked against the live pages. `LICENSES/DATA_LICENSES.md` gives the commits and the pages to check.

| dataset | licence | use |
|---|---|---|
| AgentAbstain, Liu et al.: data on Hugging Face (`antiquality/agentabstain`) | CC-BY-4.0 | the 263 act/abstain pairs. Adapted here (state shown as tables, tools and results not carried); not redistributed |
| AgentAbstain code (`AntiQuality/agentabstain`) | MIT | pair index only |
| tau2-bench (Sierra Research) | MIT | Tier 2 task data, policy text and tool code, imported unmodified |
| CRM-Agent-Bench (Salesforce) | Apache-2.0 | Tier 2 payments slice, imported unmodified |

The repository contains no dataset files. Adapted case files are generated locally and stay out of version control (`work/`, `data/`). Results computed on adapted cases are not benchmark numbers of the original projects.
