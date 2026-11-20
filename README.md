# Agent Detention

**Question.** When has an AI agent earned the right to act alone? Each agent faces the same cases as its peers and must decide whether to act or hold; when it fails it writes a root-cause analysis before it may retry; a held-back check, not its own grader, decides whether it improved. The open research question is whether a structured root-cause review helps an agent recover better than a plain retry or free-form reflection, and whether the agent reports honestly what it actually did.

**Finding so far.** The Detention step itself has not been tested: the first public-benchmark act-or-hold cases proved too thin for it, and it moves to larger multi-step benchmarks. What has been measured is the grading around it. On the same public act-or-hold cases, Claude Sonnet 5.5 scored 34 of 42 episodes correct (81%) and Claude Haiku 5.5 22 of 41 (54%), strictly graded, in the first block of a pre-registered comparison (83 of 380 planned episodes; the second block stopped on a usage limit). Haiku stopped when it should have acted on half of the act versions (10 of 20), against 2 of 21 for Sonnet. Neither misreported: 0 of 83 reports disagreed with the end state (95% upper bound 4.3%). The earlier work found and fixed two process failures that matter for the numbers: an answer leak in the case text, and a leak audit that checked nothing (see the changelog).

## Chapters

| # | chapter | question | finding | status |
|---|---|---|---|---|
| 01 | [instrument](chapters/01-instrument) | Can agents be graded consistently enough to compare them? | Built and sealed: 64 failure mechanisms, 1,147 tests passed (4 environmental failures). Core code only | partial |
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
src/agent_detention/      the case suite (Tier 1 AgentAbstain, Tier 2 tau2-bench and CRM), audit, row generator
chapters/NN-slug/         one folder per experiment: README, entry scripts, results/
tests/                    suite tests (synthetic fixture; some need the datasets or the harness tree)
scripts/fetch_data.py     downloads the public datasets at the commits the results used
```

## The harness tree

To run chapters 03 to 06, point `AD_TREE` at a copy of the harness tree, patched with the four-file change in `src/agent_detention/suite/harness_patch/`:

```
python -m agent_detention.suite.harness_patch.make_patched_copy SOURCE_TREE DEST_TREE
```

Chapter 01 publishes the self-contained core of that tree. See the chapter READMEs for what was and was not verified without it.

## How to run

Chapters 03 to 06 run on the harness tree (the VERA code: world, Detention protocol, grid runner), which this repository does not contain; see "The harness tree" above. Without it, what runs here is: chapter 01 tests (59), `tests/suite` on the synthetic fixture, `scripts/fetch_data.py`.

With the harness tree:

```
pip install -r requirements.txt
pip install -e .
python scripts/fetch_data.py --dest data
export AGENTABSTAIN_DATA=data/agentabstain-data AGENTABSTAIN_REPO=data/agentabstain
export TAU2_ROOT=data/tau2-bench CRM_BENCH_ROOT=data/crm-agent-bench
export AD_WORK=work AD_TREE=<patched harness tree>
python chapters/03-public-benchmark-suite/build_tier1_records.py --out work/tier1/tier1_set.json
python chapters/04-offline-control-gate/compat_test.py --refs eligible20 --out work/gate.md
```

No step calls a model or needs a network after the download.

## Licence and dataset credits

Code: MIT, Copyright (c) 2026 Mohammad Walid.

| dataset | licence | use |
|---|---|---|
| AgentAbstain, Liu et al.: data on Hugging Face (`antiquality/agentabstain`) | CC-BY-4.0 | the 263 act/abstain pairs. Adapted here (state shown as tables, tools and results not carried); not redistributed |
| AgentAbstain code (`AntiQuality/agentabstain`) | MIT | pair index only |
| tau2-bench (Sierra Research) | MIT | Tier 2 task data, policy text and tool code, imported unmodified |
| CRM-Agent-Bench (Salesforce) | Apache-2.0 | Tier 2 payments slice, imported unmodified |

The repository contains no dataset files. Adapted case files are generated locally and stay out of version control (`work/`, `data/`). Results computed on adapted cases are not benchmark numbers of the original projects.
