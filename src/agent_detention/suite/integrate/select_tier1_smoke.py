"""The 10 Tier 1 pairs for the live smoke (gate item 3): taken from the eligible20 sample, fixed before any live call.

Rule: the first pair (in the sample's rank order) of each of the 8 scenarios, plus the second pair of the first two
scenarios (alphabetical) that have three pairs in the sample.
"""

from __future__ import annotations

import json
import sys

from agent_detention import paths
from agent_detention.suite.tier1 import index


def main() -> int:
    pairs = index.load_index()
    records = json.loads((paths.tier1_records()).read_text())["mechanisms"]
    ok = {r["provenance"]["scenario"] + "/" + r["provenance"]["task_id"] for r in records}
    sample = index.eligible_sample(pairs, ok)
    by_cat: dict[str, list] = {}
    for p in sample:
        by_cat.setdefault(p.category, []).append(p)
    chosen = [v[0] for _, v in sorted(by_cat.items())]
    extra = [v[1] for _, v in sorted(by_cat.items()) if len(v) >= 3][:2]
    pick = chosen + extra
    ids = [f"AA-{p.category}-{p.task_id}" for p in pick]
    out = paths.reports() / "tier1_smoke10"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ids.json").write_text(json.dumps(ids, indent=1) + "\n")
    print(len(ids), *ids, sep="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
