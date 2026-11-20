"""Reduce run audits (output of ``agent_detention.suite.integrate.audit_run --json``) to a publishable summary.

    python chapters/05-live-smoke-run/summarise_audits.py --audit first_run=AUDIT.json --audit rerun=AUDIT.json --out results/summary.json

Keeps counts and verdicts. Drops route identifiers, episode ids, log file names and error text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TOP_KEYS = ("calls", "main_calls", "detention_calls", "m2_calls_checked", "m2_unattributed_calls", "m3_calls_checked",
            "m3_exercised", "ok_m1", "ok_m2", "ok_m3", "ok_no_errors", "refusal_calls", "result_files", "episodes_with_log")


def reduce_audit(audit: dict) -> dict:
    detention = {k: v for k, v in audit["detention"].items() if k not in ("route_errors", "route_refusals")}
    detention["route_error_count"] = sum((audit["detention"].get("route_errors") or {}).values())
    detention["route_refusal_count"] = sum((audit["detention"].get("route_refusals") or {}).values())
    out = {k: audit[k] for k in TOP_KEYS}
    out["leak_hits"] = len(audit["leaks"])
    out["leak_hits_distinct_episodes"] = len({x["episode"] for x in audit["leaks"]})
    out["leak_kinds"] = sorted({x["what"] for x in audit["leaks"]})
    out["call_error_count"] = len(audit["call_errors"])
    out["detention"] = detention
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="append", required=True, help="NAME=audit.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = {}
    for spec in args.audit:
        name, _, path = spec.partition("=")
        result[name] = reduce_audit(json.loads(Path(path).read_text()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
