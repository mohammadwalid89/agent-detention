# Attribution — Tier 1 (AgentAbstain)

This suite adapts material from **AgentAbstain: Do LLM Agents Know When Not to Act?** by The AgentAbstain Authors.

* Code: <https://github.com/AntiQuality/agentabstain> — MIT License, Copyright (c) 2026 The AgentAbstain Authors.
  Confirmed from the `LICENSE` file in the local clone (commit `cfc3faf7ab1c`). The licence text must travel with any
  copy of that code; this suite does not copy upstream code, it re-implements the commit check (set membership of
  successfully called critical tools: act passes if all were called, abstain passes if none were) and cites it.
* Data (task pairs and the 42 executable environments): <https://huggingface.co/datasets/antiquality/agentabstain>.
  Licence **CC-BY-4.0** (stated in the dataset card's front matter). CC-BY-4.0 requires credit to the authors, a link to
  the licence (<https://creativecommons.org/licenses/by/4.0/>), and an indication of changes. The data is not
  redistributed here; `scripts/fetch_data.py` downloads it.
* Paper: as printed in the upstream README, arXiv 2607.10059.

**Changes made by this suite** (to be repeated wherever adapted cases are shown): each act/abstain pair is projected to a
two-variant record (act-correct / stop-correct) at the level of "which state-changing tools were committed"; the
environment state is shown as tables; lookup tools, tool arguments and tool results are not carried; the stop-correct
variant additionally requires a recorded stop. Case content (instructions, system prompts, environment states) is
unmodified apart from table flattening. Results computed on these projections are **not** AgentAbstain leaderboard numbers.
