# 02. Provisional pilots on small models

**Question.** Does the instrument run end to end on real models, and do the control arms score where they must?

**Setup.** Two provisional pilots of 432 episodes each: four fixed control policies (always act, act-first heuristic, uniform random, abstain first) and two small free hosted models, 72 episodes per arm. $0 spent. Every arm ran through the full harness, and every episode was counted, including model failures.

**Result.** Pass counts out of 72 in pilot 1 (`results/summary.json`, key `pilot1`): act-first heuristic 36, uniform random 29, gpt-oss-20b 21 (arm `MODEL_1`), Nemotron 30B 16 (arm `MODEL_2`), always act 9, abstain first 0. Model failures: 5 of 72 episodes for gpt-oss-20b (6.9%) and 22 of 72 for Nemotron (30.6%). No harness faults in pilot 1; every model failure was counted. Pilot 2 repeated the pattern: gpt-oss-20b 23 passes and 4 model failures, Nemotron 18 and 24 (key `pilot2`; one gpt-oss-20b episode there carries a harness-fault flag and is scored INVALID).

Why the models scored below a heuristic: the prompt listed `delegate` as a permitted action, the models answered `{"action": "delegate"}`, and the harness scored it as an unknown action. Of 62 such steps, 26 were `delegate`, 24 `consult` and 12 `observe` (from the independent review's code audit, not from this folder).

**What it means.** The pipeline ran and the control arms landed where they should. The models scoring below a simple heuristic says more about the setup than about agents, so these are provisional results on small models only. An independent review the next day found the instrument had never seen a frontier model (see the root CHANGELOG).

**How to run.** The pilot runner belongs to the harness tree and is not published. What is published is the reduction from a pilot result file to the per-arm counts:
```
python chapters/02-small-model-pilots/summarise_pilot.py --pilot pilot1=PATH --pilot pilot2=PATH --out results/summary.json
```

**Status.** Partial: results only plus the summariser; the runner and the model routes are not published. Provisional. `summarise_pilot.py` was run on the two real pilot result files to produce `results/summary.json` (the counts match the figures above); `py_compile` and `ruff check` clean. No model or network call.
