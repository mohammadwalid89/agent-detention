"""Reduce the scorer's output to the aggregates that back the published numbers.

    python chapters/06-stage0-side-by-side/summary_from_scores.py --scores scores_stage0.json --out results/summary.json

Keeps the primary-sample aggregates and the coverage counts. Drops per-episode records, the failure list,
the not-run list and the second-reader material (these hold report text and case ids).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    scores = json.loads(Path(args.scores).read_text())
    summary = {
        "episodes_scored": scores["episodes"],
        "not_run_rows": len(scores["not_run"]),
        "interrupted_episodes": len(scores["interrupted"]),
        "unreadable_files": len(scores["unreadable_files"]),
        "primary": scores["primary"],
        "attempt_then_stop_for_second_reader": len(scores["attempt_then_stop_for_second_reader"]),
        "overclaims": len(scores["overclaims"]),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
