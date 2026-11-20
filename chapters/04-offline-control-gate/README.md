# 04. Offline control gate

**Question.** Does the grader score obvious policies the way it must before any model is tested?

**Setup.** Fixed policies, run through the full pipeline with no model: always act, always hold, do nothing, and a coin flip between act and hold (20 seeds). A uniform-random action policy is reported as a descriptive extra. Tier 1 sample: 20 adaptable pairs, drawn by the same seeded, stratified rule as the 20 pairs fixed in advance but restricted to pairs that can be adapted. Of the 20 pairs fixed in advance, 13 were read-only and could not be adapted (7 adapt), so both samples were run and both are reported. Tier 2: 10 pairs chosen by a rule fixed before running.

**Result.** From `results/tier1_eligible20.md` (20 adaptable pairs; act-version pass rate / hold-version pass rate): always-act 100% / 0%; always-hold 0% / 100%; no-op 0% / 0%; coin flip 48.3% overall (acceptance band 45 to 55%); uniform random 47.1% overall. On the 7 adaptable pairs of the pre-registered 20 (`results/tier1_prereg20.md`) the pattern is the same and the coin lands at 47.5%. Tier 2 (`results/tier2_dryrun_summary.json`, 10 pairs, 20 instances): all five control checks 100%, no problem pairs.

One disagreement between the end-state check and the runner's own verdict is understood and reported: a forbidden commit that the world refuses, followed by a stop, passes the runner's `valid_disposition` but fails the strict check (155 of 800 uniform-random episodes, 0 of 800 coin episodes in the 20-pair sample). Any unexplained disagreement would exclude the pair; there were none.

The leak: the escalation row of the first adapted cases carried `critical_tool_failure/preview_019`, the scenario name and task id. It was found by reading a prompt by hand, not by the automated scan, and replaced by an opaque `case-<hash>`; everything was rebuilt and re-run. Chapter 05 explains why the automated scan had not caught it.

**What it means.** The grader passes its own sanity checks: policies that should win or lose do, and the coin flip lands near half. This says nothing about agents. It shows the end-state check and the runner's verdict agree except in one understood case, which matters for how Stage 0 scores holds (chapter 06).

**How to run.**
```
export AD_TREE=<patched harness tree> AGENTABSTAIN_DATA=... AGENTABSTAIN_REPO=... AD_WORK=work
python chapters/04-offline-control-gate/compat_test.py --refs eligible20 --out gate.md --json gate.json
python chapters/04-offline-control-gate/compat_test.py --synthetic --out synthetic.md    # no dataset needed
```
`--refs` takes `sample20` (the pre-registered 20), `eligible20` or `all` (all 131 adaptable pairs).

**Status.** Done. Code cleaned from the run version: paths from environment variables, the sealed-tree default removed. Verified here, no model call: `--synthetic` ran on a generated fixture (24 pairs, 4 seeds; controls behave as expected; the fixture is stamped SYNTHETIC and is not a result); `--refs eligible20` and `--refs sample20` were re-run on local dataset clones and reproduce the figures above (20 of 20 kept, coin 48.3%; 7 of 20 kept, coin 47.5%). The Tier 2 figures are copied from the build report's dry run, which was not re-run. `py_compile` and `ruff check` clean.
