# Attribution (Tier 2)

The case files this code builds are **derived works**: they contain material from the datasets below (task text, DB
rows, policy sentences, tool-computed end states), reshaped into the Agent Detention mechanism format. They are
generated locally by `build_tier2.py` and are not part of this repository. Licence texts travel with the clones that
`scripts/fetch_data.py` downloads.

| dataset | licence | copyright | what is used |
|---|---|---|---|
| **tau2-bench** (airline, retail, telecom) | MIT | 2025 Sierra Research | task statements (the customer's request text), the domain databases, the policy sentences (quoted with file and line), and tau2's own tool code, which is imported unmodified to replay gold actions. Five small third-party packages tau2 imports (loguru, docstring_parser, deepdiff, addict, toml) are replaced by stubs in `suite/_shims/`; the stubs never touch tau2's tool logic |
| **CRM-Agent-Bench** (payments) | Apache-2.0 | 2026 Salesforce, Inc. | the payments task catalogue (6 stop tasks and 4 act tasks used), the payments policy text, and the payments tool environment (imported unmodified). Its NOTICE file lists Apache Burr (Apache-2.0), which is not used |

Changes made (Apache-2.0 section 4(b)): CRM tasks are re-expressed as two-variant records; two of the six pairs have an
act variant produced by one edit (marked `origin: derived`, edit listed in `suite.edit`).
No dataset used here is non-commercial. Cite the tau2-bench and CRM-Agent-Bench papers in any publication of results.
