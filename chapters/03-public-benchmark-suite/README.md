# 03. Public benchmark suite

**Question.** Can public agent benchmarks be turned into act-or-hold pairs the instrument can grade?

**Setup.** Tier 1: AgentAbstain (data CC-BY-4.0), 263 pairs in eight scenarios. Tier 2: tau2-bench (MIT) and a CRM payments slice (CRM-Agent-Bench, Apache-2.0), 133 pairs. Each pair has an act version and a hold version of the same task. Operational pairs were adapted; read-only informational pairs were excluded because there is nothing to commit. Case text for Tier 2 is generated from source fields and states the governing policy sentence; it is not a quotation. See `src/agent_detention/suite/tier1/ATTRIBUTION.md` and `tier2/ATTRIBUTION.md`.

**Result.** 131 of 263 Tier 1 pairs adapted and 132 excluded (all informational). With Tier 2: 264 pairs, 528 instances, none rejected. A solvability scan in code (no model calls) then left 95 of the 131 Tier 1 pairs usable for Stage 0: 19 act versions demand a tool the agent is never offered, and in 34 pairs both versions show the model the same prompt. Two of eight scenarios (`critical_tool_failure`, `insufficient_tool_capability`) have no usable pair until they are repaired. All in `results/summary.json`.

| scenario | pairs adapted | usable after the scan |
|---|---:|---:|
| ambiguous_action_specification | 15 | 15 |
| conflicting_constraints | 15 | 14 |
| conflicting_evidence | 15 | 14 |
| critical_tool_failure | 15 | 0 |
| emergent_risk_discovery | 17 | 17 |
| high_stakes_action | 15 | 15 |
| insufficient_tool_capability | 19 | 0 |
| missing_critical_parameter | 20 | 20 |

**What it means.** The benchmarks can be projected into a graded act-or-hold form, but the projection removes information: lookup tools, tool arguments and real side effects are not carried, so results are not leaderboard-comparable. The read-only half of AgentAbstain (do not over-abstain) is not covered. The Tier 2 half is easy to guess, because its case text states the rule and random play passes most hold versions; Tier 1 and Tier 2 are reported separately. 69% of Tier 2 pairs rest on two retail status rules.

**How to run.**
```
python scripts/fetch_data.py --dest data
export AGENTABSTAIN_DATA=data/agentabstain-data AGENTABSTAIN_REPO=data/agentabstain
export TAU2_ROOT=data/tau2-bench CRM_BENCH_ROOT=data/crm-agent-bench AD_WORK=work AD_TREE=<patched harness tree>
python chapters/03-public-benchmark-suite/build_tier1_records.py --out work/tier1/tier1_set.json
python -m agent_detention.suite.tier2.build_tier2
python chapters/03-public-benchmark-suite/scan_solvability.py --out work/solvability.json
```
The harness tree is made from its source tree with `python -m agent_detention.suite.harness_patch.make_patched_copy SRC DEST`.

**Status.** Done (code cleaned from the run version: paths and dataset locations now come from environment variables; the smoke-run corroboration block of the scan, which read run transcripts, and the manifest writers were removed). Verified here against local clones of the datasets, read-only, no network and no model call: `build_tier1_records.py` adapted 131 of 263 pairs with 132 excluded, record set hash prefix `ec42f8a6` as in the run; `scan_solvability.py` gave 108 solvable, 19 unsolvable, 4 unclear, 34 indistinguishable pairs and 95 usable; its 10 known-answer tests pass. `build_tier2` produced 133 pairs from the local tau2-bench and CRM clones and the 400 Tier 2 tests pass; the 25 tests under `tests/suite` pass against the patched harness tree. `py_compile` and `ruff check` clean. The harness tree these ran against is the unpublished one (see the root README).
