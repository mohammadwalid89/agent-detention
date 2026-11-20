# 05. First live smoke run

**Question.** Does the full pipeline run end to end with a live model?

**Setup.** A small smoke set through the grid runner with one Claude Sonnet route, Detention phase included. Detention is the step where a failed episode is reviewed, a lesson is written, and sibling cases are rerun with that lesson (four arms: structured review, plain retry, a compute-matched free-form review, scripted elimination). Calls were capped at 60 per run.

**Result.** From the run audits (`results/summary.json`, `first_run_audit_after_rebuild` and `rerun_audit`) and the compatibility record:

- First run: the main phase ran 20 episodes (10 pairs) in 49 calls; all 10 hold versions and 9 of 10 act versions were graded correct, and no report disagreed with the end state. The Detention phase made 11 more calls and stopped early at the 60-call cap; one arm was refused by the provider's safeguards and one was interrupted. The first audit could not fail, because it matched no calls to cases (0 of 49 examined). The corrected audit then found the scenario name, task id and variant label in the Detention prompts: the failed episode's id, which the row generator had built from them, was copied into the review and retry prompts. Both were fixed (episode ids became opaque hashes) before the re-run.
- Re-run: 5 pairs, chosen by a rule fixed beforehand so a full Detention cycle fit the cap. 41 calls: 22 in the main phase and 19 in Detention. All 5 hold versions and 4 of 5 act versions were graded correct; the one failure was the same case as in the first run. The Detention cycle completed all four arm rows with no refusals and no errors. The rebuilt audit matched all 41 calls to their cases and found no hidden answer material and no arm labels in any prompt (audit counts: 41 of 41 calls checked for leaks, 19 of 19 Detention calls checked for arm labels).
- Two earlier attempts were refused by the runner's own checks (a wrong slot for the model; a one-field fingerprint mismatch in the isolation check).

**One failed episode and a thrown-out root cause.** The task said to mark counterfeit items only after verifying them with a tool, and the adapted case did not offer that tool. The agent escalated, quoting the missing tool, and its report matched the end state; the grade was a failure because the case, not the agent, was at fault (chapter 03's scan later found 19 such act versions). The structured review named the cause `insufficient_tool_capability`, which was the scenario label leaked into its prompt; the audit caught it, and that root cause was discarded.

**What it means.** The pipeline runs end to end and the audit now checks every call. A smoke run tests plumbing, not agent quality, and says nothing yet about agents. The safeguard-refusal path did not occur in the re-run; it is covered by offline tests only. Detention in the re-run rests on one failed episode.

**How to run.** Offline, with mock adapters (no model call):
```
export AD_TREE=<patched harness tree> AGENTABSTAIN_DATA=... AGENTABSTAIN_REPO=... AD_WORK=work
python -m agent_detention.suite.integrate.build_inputs
python -m agent_detention.suite.integrate.select_tier1_smoke
python -m agent_detention.suite.integrate.rows --ids-file work/reports/tier1_smoke10/ids.json --slot O1 --repeats 1 --out work/reports/rows.json
python -m agent_detention.suite.integrate.run_suite --assignment work/reports/rows.json --out work/rundry --dry-run --detention
python -m agent_detention.suite.integrate.audit_run work/rundry --json audit.json
python chapters/05-live-smoke-run/summarise_audits.py --audit rerun=audit.json --out summary.json
```
A live run needs the harness tree's model routes, a signed pilot gate and a preflight report; those scripts reveal how the author's machine was set up and are not published. Nothing in this repository calls a model.

**Status.** Done (clean re-run passed). Code cleaned from the run version: `audit_run.py`, `build_inputs.py`, `binding.py`, `rows.py`, `run_suite.py` and the selectors, with paths from environment variables. Not published: the gate writer, route preflight and fingerprint scripts. Verified here, no model call: the mock-adapter dry run above completed 20 of 20 rows (45 mock calls) and its audit passed M1 and M2 with 45 of 45 calls checked (M3 is unexercised on mock runs without a failed episode, as in the source dry run); the 25 tests under `tests/suite` pass, including the audit tests that pin that the audit fails when it checks nothing. `results/summary.json` was produced with `summarise_audits.py` from the two real audit files. The live runs themselves were not repeated.
