"""Derive the Stage 0 sample from the solvability scan (deterministic; no model calls).

    python chapters/06-stage0-side-by-side/make_sample.py --scan solvability.json --out stage0_sample.json

primary  = act verdict SOLVABLE and versions distinguishable (scan 'eligible')
flagged  = act verdict UNCLEAR and versions distinguishable (run as a separate stratum, never in a primary number)
excluded = everything else, with the scan's reason codes
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True, help="solvability.json from chapter 03")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    scan_path = Path(args.scan)
    scan = json.loads(scan_path.read_text())
    primary, flagged, excluded = [], [], []
    for c in scan["cases"]:
        p1 = c["pair"]["indistinguishable"]
        if c["verdict"] == "SOLVABLE" and not p1:
            primary.append(c["mechanism_id"])
        elif c["verdict"] == "UNCLEAR" and not p1:
            flagged.append({"pair": c["mechanism_id"], "codes": c["codes"]})
        else:
            excluded.append({"pair": c["mechanism_id"], "act_verdict": c["verdict"], "codes": c["codes"],
                             "P1_versions_indistinguishable": p1})
    by_scen = collections.Counter(m.split("-")[1] for m in primary)
    body = {
        "derived_from": {"file": scan_path.name, "sha256": hashlib.sha256(scan_path.read_bytes()).hexdigest(),
                         "scan_version": scan["scan_version"], "record_set_sha256": scan["inputs"]["record_set_sha256"]},
        "primary_pairs": sorted(primary), "primary_n": len(primary),
        "primary_by_scenario": dict(sorted(by_scen.items())),
        "flagged_pairs": flagged, "excluded_pairs": excluded,
        "episodes_per_model": {"primary": 2 * len(primary), "with_flagged": 2 * (len(primary) + len(flagged))},
    }
    body["primary_ids_sha256"] = hashlib.sha256("\n".join(sorted(primary)).encode()).hexdigest()
    Path(args.out).write_text(json.dumps(body, indent=1) + "\n")
    summary = {k: body[k] for k in ("primary_n", "primary_by_scenario", "episodes_per_model", "primary_ids_sha256")}
    print(json.dumps(summary, indent=1))
    print("flagged:", [f["pair"] for f in flagged], "excluded:", len(excluded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
