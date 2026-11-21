# Dataset and code licences

Record of the licence of every third-party dataset and code base this repository uses, the official page it should be checked against, and what was checked.

**Check of 2026-11-20.** The official pages were to be fetched with `curl`. The fetch could not run: outbound connections were refused by the network policy of the build environment, so **no licence below was verified against its live page.** Each one was instead read from the licence file or dataset card in the local clone the results were computed from, at the commit given. That shows the licence at that commit; it does not show the page today.

No dataset file is in this repository. `scripts/fetch_data.py` downloads each source at the commit below; case files adapted from them are generated locally into `work/` and are not committed.

| Source | Used for | Official page to check | Licence as stated in this repository | Local evidence (commit) | Status (2026-11-20) |
|---|---|---|---|---|---|
| AgentAbstain data (Hugging Face `antiquality/agentabstain`) | Tier 1 act/abstain pairs, adapted | https://huggingface.co/datasets/antiquality/agentabstain | CC-BY-4.0 | dataset card front matter `license: cc-by-4.0` (`842228426c2a`, 2026-07-17) | Not verified against the live page; matches the local card. |
| AgentAbstain code (GitHub `AntiQuality/agentabstain`) | pair index only | https://github.com/AntiQuality/agentabstain | MIT | `LICENSE`: MIT License, The AgentAbstain Authors (`cfc3faf7ab1c`, 2026-09-28) | Not verified against the live page; matches the local file. |
| tau2-bench (Sierra Research) | Tier 2 task data, policy text and tool code, imported unmodified | https://github.com/sierra-research/tau2-bench | MIT | `LICENSE`: MIT License, Sierra Research (`c06b07dd7c4d`, 2026-11-10) | Not verified against the live page; matches the local file. |
| CRM-Agent-Bench (Salesforce) | Tier 2 payments slice, imported unmodified | https://github.com/salesforce-misc/crm-agent-bench | Apache-2.0 | `LICENSE.txt`: Apache License Version 2.0, Salesforce, Inc. (`fc9575cf9bda`, 2026-11-02) | Not verified against the live page; matches the local file. |

`src/agent_detention/suite/tier1/ATTRIBUTION.md` and `suite/tier2/ATTRIBUTION.md` give the attribution each licence asks for. The harness tree (`harness/`) vendors no third-party code.

## To finish the check

On a machine with network access, open each page above (for example `curl -sL <url>`), confirm the licence, and replace "Not verified" with the licence found and the date.
