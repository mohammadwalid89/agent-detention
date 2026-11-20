"""Adapt ALL Tier 1 pairs (needs the dataset snapshot) into one manifest in the harness's record format.

    AGENTABSTAIN_DATA=<snapshot> AGENTABSTAIN_REPO=<clone> python chapters/03-public-benchmark-suite/build_tier1_records.py \\
        --out work/tier1/tier1_set.json

Writes ``{"label", "mechanisms": [...], "excluded": [{pair, stage, code, reason}], "stats": {...}}``.
Exits 2 when the data is missing (``scripts/fetch_data.py``). ``--synthetic`` uses the fixture and refuses to
write anywhere but a path containing ``synthetic``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from agent_detention.suite.tier1 import adapter, index, loader


def observation_chars(record: dict) -> list[int]:
    """Size of the opening observation per variant (prompt-budget planning); nothing is truncated."""
    out = []
    for v in record["variants"]:
        out.append(len(json.dumps(v["entry_tables"])) + len(v["goal_text"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    if args.synthetic:
        if "synthetic" not in Path(args.out).name:
            raise SystemExit("synthetic output must have 'synthetic' in its file name")
        from agent_detention.suite.tier1 import synthetic

        root = Path(tempfile.mkdtemp(prefix="aa_syn_"))
        refs = synthetic.make(root)
    else:
        root, refs = None, index.load_index()
    records, excluded = [], []
    try:
        for ref in refs:
            try:
                records.append(adapter.adapt_pair(loader.load_pair(ref, root)))
            except loader.PairSchemaError as exc:
                excluded.append({"pair": ref.key, "stage": "load", "code": "SCHEMA", "reason": str(exc)})
            except adapter.AdaptExclusion as exc:
                excluded.append({"pair": ref.key, "stage": "adapt", "code": exc.code, "reason": exc.detail})
    except loader.DataMissing as exc:
        print(f"BLOCKED: {exc}")
        return 2
    sizes = [max(observation_chars(r)) for r in records]
    payload = {
        "label": "AgentAbstain Tier 1 adapted set" + (" (SYNTHETIC FIXTURE)" if args.synthetic else ""),
        "mechanisms": records,
        "excluded": excluded,
        "stats": {
            "requested_pairs": len(refs), "adapted_pairs": len(records), "excluded_pairs": len(excluded),
            "max_opening_chars": max(sizes) if sizes else 0,
            "median_opening_chars": sorted(sizes)[len(sizes) // 2] if sizes else 0,
            "excluded_by_code": {c: sum(e["code"] == c for e in excluded) for c in sorted({e["code"] for e in excluded})},
        },
    }
    text = json.dumps(payload, indent=1, sort_keys=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(f"wrote {args.out}: adapted {len(records)}/{len(refs)}, excluded {len(excluded)}, sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
