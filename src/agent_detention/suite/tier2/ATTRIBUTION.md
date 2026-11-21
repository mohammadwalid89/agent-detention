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

## Upstream text in this repository's source

The adapter source also contains 17 short policy sentences copied from these projects, used to locate the policy line
each case cites: 16 from tau2-bench (`tau2_adapter.py`, `tau2_native_stop.py`, `tau2_telecom.py`) and 1 from
CRM-Agent-Bench (`crm_payments.py`). Their notices:

**tau2-bench**

```
MIT License

Copyright (c) 2025 Sierra Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**CRM-Agent-Bench:** Copyright (c) 2026 Salesforce, Inc., Apache License 2.0
(<https://www.apache.org/licenses/LICENSE-2.0>).
