# Changelog

Dated history of the experiments, including what failed. The site shows findings; this file holds the process.

## 2026-11-20
- Repository assembled from the build folders of the experiments. Code cleaned from the run versions (paths and dataset locations from environment variables, unused imports removed, duplicated and infrastructure-specific scripts left out). Re-verified offline against local dataset clones: Tier 1 adaptation (131 of 263), solvability scan (95 usable), Tier 2 build (133 pairs), offline control gate (coin 48.3%), Stage 0 row files (byte-identical to the run's).
- Not published: the harness tree the runs executed (about 58,000 lines beyond the core in chapter 01), the run launcher and route configuration, and all run outputs (episode records, prompts, call logs, transcripts).
- Known issue carried over: `build_inputs --check` writes the suite source file into the harness tree before comparing and restores it afterwards; it should not write in check mode.

## 2026-11-18 to 2026-11-19 (Stage 0)
- Stage 0, block 1 ran in full: 80 episodes, 212 calls, audit clean, instrument checks passed before any accuracy was read.
- Stage 0, block 2 stopped two minutes in. The subscription's 5-hour usage limit paused both model routes; the runner had no automatic retry configured and waited for a manual resume until the launcher's 100-minute timeout killed it. 3 of 308 episodes finished, 2 were cut off mid-episode, 305 rows never ran, including all 8 flagged-stratum episodes. The cause was usage, not the instrument or the models.
- The decision rules written for the full sample (the 90% ceiling rule and the 60-failure rule) were not applied, because only 83 of 380 planned primary episodes were scored. Scoring partial data was a written deviation from the pre-registration, recorded before any outcome was read.
- An amendment approved that evening (Nov 18, Pacific time) would have resumed block 2 with an automatic 15-minute retry on usage-limit pauses and a higher call cap. The launcher's result record for that attempt carries the same values as the first block-2 attempt (return code 124, 16 calls). The published partial result is unchanged: 83 of 380.
- Of the 27 strict failures, 15 were eligible for Detention under the runner's own rule and 12 were not (11 attempt-then-stop holds that the runner counts as passes, and 1 contained-only outcome). Attempt-then-stop reports (11) await a second human reader.

## 2026-11-18
- Public benchmark suite built: Tier 1 (AgentAbstain) 263 pairs, 131 adapted, 132 excluded as read-only; Tier 2 (tau2-bench, CRM payments slice) 133 pairs. ERP-Bench (Tier 3) was inventoried but not run: it needs containers and a network.
- The 20 pairs fixed in advance for the compatibility gate turned out to be mostly read-only: 7 adapt, 13 do not. A second sample drawn by the same seeded rule from adaptable pairs only was defined before any control or live run, and both were run.
- Offline control gate passed: always-act 100/0, always-hold 0/100, no-op 0/0, coin 48.3%; Tier 2 all five control checks 100%.
- The case text leaked the answer: the escalation row carried the scenario name and task id (`critical_tool_failure/preview_019`). Found by reading a prompt by hand, fixed with an opaque case id, everything rebuilt and re-run.
- The live smoke could not start at first: the sandbox that built the suite had no route to the model provider, so the live step was left blocked and recorded as not run.
- Live smoke attempts, in order. The first live attempts were refused by the runner's own checks: the rows named the wrong model slot, and the preflight's isolation fingerprint differed from the runner's in one field (the isolation root was typed by hand in one script and absent from the runner config). Both fixed. First full run: 20 episodes, 49 calls, clean main phase; the Detention phase was cut off by the 60-call cap, one arm refused by the provider's safeguards and one interrupted. The refusal had been recorded as a harness fault; refusal handling (recognise the provider's wording, retry once, count refusals apart from errors) was then implemented and covered by offline tests, and was not seen live again.
- The leak audit checked nothing. An agent reported the smoke audit clean and said its scan had found the earlier leak. The audit attributed calls to cases through a field the call records do not carry, so it examined 0 of 49 calls and could not fail; the leak had been found by reading a prompt. The agent wrote the correction itself; the audit was rebuilt to attribute calls through the run's assignment table, to report how many calls it scanned, and to fail if that is zero (with tests).
- The rebuilt audit then found a real leak path: Detention review and retry prompts copied the failed episode's id, which carried the scenario name, task id and the act-version suffix. Episode ids became opaque hashes.
- A root cause written in the first Detention cycle named `insufficient_tool_capability`, which was the leaked scenario label. It was discarded.
- Re-run with 5 pairs (chosen by a rule fixed beforehand so a full Detention cycle fit the 60-call cap): 41 calls, all four Detention arms completed, audit clean with 41 of 41 calls and 19 of 19 Detention calls checked.
- Solvability scan, no model calls: 19 act versions demand a tool the agent is never offered, and 34 pairs show the model the same prompt in both versions; 95 of 131 pairs usable, two of eight scenarios with none. A first version of the scan had flagged numbers stated in the request itself and equal shares as unclear (eligible pairs 92); the second version fixed both (95).
- Four tests in the harness tree fail in the build sandbox for environmental reasons and fail identically on the sealed tree: a fixed temporary path, a tokenizer download, and a user-id check. The count was 3 before one test started to fail under the sandbox's user mapping.

## 2026-11-16
- Independent review of the instrument before a main run: not ready. No frontier model had run an episode. In a 40-episode timing run, 25 episodes tried to delegate and all 25 handoffs were rejected because the prompt listed an action the validator refused. The world was a single ticket row. A follow-up code audit found that the reporting turn printed the answers the misreport measure tests, that condition labels and Detention arm labels were visible in prompts, and that the prompt text manufactured invalid actions (62 steps scored as unknown actions: 26 delegate, 24 consult, 12 observe). Decision: rebuild the cases on public benchmarks and run a small frontier smoke before any main grid.

## 2026-11-14
- Two provisional pilots of 432 episodes each on two small free hosted models and four control policies, $0. No harness faults in pilot 1; model failures up to 30.6% for one route.

## 2026-11-12 to 2026-11-14
- Instrument built: 64 mechanisms (48 development, 16 in two sealed families), signed scoring, pre-registered Detention protocol. The 1,147-passed, 4-environmental-failures test count is from the Nov 18 run of the tree.

## Not run
- An experiment on the AppWorld benchmark (does a reviewed failure prevent similar failures on sibling tasks?) exists as a short development script and one development scenario. It is not part of this repository; the Detention question is still open.
