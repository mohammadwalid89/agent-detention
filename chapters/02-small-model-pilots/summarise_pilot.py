"""Per-arm pass counts from a pilot result file (the 432-episode record the harness writes).

    python chapters/02-small-model-pilots/summarise_pilot.py --pilot pilot1=PATH --pilot pilot2=PATH --out results/summary.json

Reads ``episodes`` (one record per episode) and counts, per arm: episodes, PASS, FAIL, INVALID, episodes that hit a
model failure, episodes that ended on one, and episodes flagged as a harness fault. Only these counts are written;
no episode content, prompts or model ids beyond the arm label.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def arm_label(arm: str) -> str:
    match = re.match(r"MODEL:M(\d+):", arm)
    return f"MODEL_{match.group(1)}" if match else arm


def summarise(path: Path) -> dict:
    episodes = json.loads(path.read_text())["episodes"]
    arms: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for e in episodes:
        c = arms[arm_label(e["arm"])]
        c["episodes"] += 1
        c[e["execution_outcome"]] += 1
        c["model_failed"] += bool(e["model_failed"])
        c["ended_on_model_failure"] += bool(e["ended_on_model_failure"])
        c["harness_fault"] += bool(e["harness_fault"])
    out = {}
    for arm, c in sorted(arms.items()):
        row = {k: c.get(k, 0) for k in ("episodes", "PASS", "FAIL", "INVALID", "model_failed", "ended_on_model_failure", "harness_fault")}
        row["model_failure_rate_pct"] = round(100 * row["model_failed"] / row["episodes"], 1)
        out[arm] = row
    return {"episodes": len(episodes), "arms": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="append", required=True, help="NAME=pilot_result.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = {}
    for spec in args.pilot:
        name, _, path = spec.partition("=")
        result[name] = summarise(Path(path))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
