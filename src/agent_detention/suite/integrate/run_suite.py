"""Run suite rows through the unchanged ``GridRunner`` (dry run with mocks, or the pilot scope live).

    python -m agent_detention.suite.integrate.run_suite --assignment rows.json --out RUNDIR --dry-run [--detention]
    python -m agent_detention.suite.integrate.run_suite --assignment rows.json --out RUNDIR --pilot \
        --pilot-gate gate.json --preflight-acceptance report.json --config config.json \
        --signing-key-file KEYFILE --detention

Mirrors ``runner_v13.cmd_runner`` except that the source binding is the suite binding (``binding.py``) instead of the
set-12 one.  Everything else (gate validation, pilot scope checks, preflight acceptance, transcripts, solo mode,
Detention v14 with sibling transfer) is the tree's own code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_detention.suite.integrate.binding import (
    SuiteBindingError,
    resolve_suite_binding,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot-gate", default=None)
    ap.add_argument("--preflight-acceptance", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--signing-key-file", default=None)
    ap.add_argument("--detention", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    if args.dry_run == args.pilot:
        raise SystemExit("choose exactly one of --dry-run / --pilot")

    from vera.delegation import runner_v13 as r
    from vera.delegation.routes.registry import load_registry

    try:
        binding = resolve_suite_binding()
        registry = load_registry(profile="claude_only")
        gate = {}
        if args.pilot:
            if not (args.pilot_gate and args.preflight_acceptance and args.config):
                raise r.StageGateError("--pilot needs --pilot-gate, --preflight-acceptance and --config")
            gate = r.validate_pilot_gate(args.pilot_gate)
        config = r.load_runner_config(args.config, registry=registry, dry_run=args.dry_run)
        rows, meta = r.load_assignment_rows(args.assignment, scope="pilot" if args.pilot else "dry_run")
        key = None
        if args.signing_key_file:
            key = Path(args.signing_key_file).read_text().strip().encode()
        runner = r.GridRunner(
            rows, config=config, workdir=args.out, registry=registry, dry_run=args.dry_run, gate=gate,
            max_episodes=args.limit, signing_key=key, require_measured_routes=args.pilot,
            preflight_acceptance=args.preflight_acceptance, source_binding=binding, assignment_meta=meta,
            record_transcripts=True, detention_protocol="v14", solo=True,
        )
        summary = runner.run()
        detention = runner.run_detention_phase() if args.detention else None
    except SuiteBindingError as exc:
        print(json.dumps({"status": "REFUSED_SOURCE_BINDING", "reason": str(exc)}))
        return 9
    except r.StageGateError as exc:
        print(json.dumps({"status": "REFUSED_STAGE_GATE", "reason": str(exc)}))
        return 4
    except r.RunnerConfigError as exc:
        print(json.dumps({"status": "REFUSED_CONFIG", "reason": str(exc)}))
        return 5
    out = {k: summary.get(k) for k in ("status", "run_id", "run_scope", "rows_assigned", "rows_completed",
                                       "rows_remaining", "model_calls_total")}
    if detention is not None:
        out["detention"] = {k: detention.get(k) for k in ("status", "selected_episodes", "rows_completed")}
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0 if summary["status"] in ("completed", "completed_limited") else 3


if __name__ == "__main__":
    sys.exit(main())
