"""Audit one run directory against the compatibility smoke's mechanical checks (works on mock and live runs).

    python -m agent_detention.suite.integrate.audit_run RUNDIR [--json out.json]

M1  every call record in ``episode_calls/*.jsonl`` (main) and ``detention/episode_calls/*.jsonl`` carries ``messages`` and
    ``reply_text``; at least one call must exist.
M2  no answer leakage: no model-visible message contains text that is only in the hidden answer material of the case the call
    belongs to (the abstention contract's rationale/prerequisite, the scenario/why strings, the scenario name and task id; in the
    main phase also the OTHER variant's goal text - a Detention sibling transfer legitimately shows the sibling's own text).  Each call
    is attributed to its case from its ``episode_id`` (``suite-<slot>-r<n>-<mechanism>-V<k>``); ``m2_calls_checked`` says how many
    calls were actually scanned and ``ok_m2`` is False if that is zero (a scan that examined nothing proves nothing).
M3  arm blinding: no model-visible Detention message carries an arm name (A_DETENTION ... E_ELIMINATION_RETRY, "arm"-style labels),
    a review-mode label (structured_valid, free_form, plain_previous_failure ...) or a condition label (NECESSARY / OPTIONAL /
    DODGE / ABSTENTION).  ``ok_m3`` is True only if Detention transcripts exist (``m3_calls_checked`` > 0) and none carries one;
    with no Detention call, M3 is unexercised and ``ok_m3`` is False (never vacuously true).
Errors: a call record or result file with a real error counts as an error.  A provider safeguard refusal (``PROVIDER_REFUSAL``) is
    counted separately (``refusal_calls``) and is not an error.  Detention rows are summarised from ``detention_summary.json``.
Exit 0 iff ok_m1, ok_m2, ok_no_errors and no M3 leak (an unexercised M3 does not by itself fail the exit code; read ``ok_m3``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from agent_detention.suite.integrate import build_inputs as bi

EPISODE_RE = re.compile(r"^suite-[^-]+-r\d+-(?P<mid>.+)-(?P<v>V[12])$")
ARM_NAMES = ("A_DETENTION", "B_PLAIN_RETRY", "C_COMPUTE_MATCHED_REVIEW", "D_SCHEMA_UNCHECKED", "E_ELIMINATION_RETRY")
REVIEW_MODES = ("structured_valid", "structured_raw_noncompliant", "structured_missing", "free_form_missing_token", "free_form",
                "plain_previous_failure", "scripted_elimination", "review_mode", "review_text_arm")
CONDITION_LABELS = ("NECESSARY", "OPTIONAL", "DODGE", "ABSTENTION")


def hidden_strings(mech: dict, variant_id: str, *, phase: str) -> dict[str, str]:
    """Hidden answer material for this case, as {label: text} (short generic strings are skipped)."""
    out: dict[str, str] = {}
    contract = mech.get("abstention_contract") or {}
    for key in ("rationale", "prerequisite"):
        if contract.get(key):
            out[f"contract.{key}"] = str(contract[key])
    for v in mech["variants"]:
        sc = v.get("scenario") or {}
        for key in ("correct_outcome", "why", "condition"):
            if sc.get(key) and len(str(sc[key])) > 12:
                out[f"{v['variant_id']}.scenario.{key}"] = str(sc[key])
        if phase == "main" and v["variant_id"] != variant_id and v.get("goal_text"):
            out["other_variant.goal_text"] = v["goal_text"]
    for v in mech["variants"]:  # the variant ids themselves: ...-V1 is act-correct, ...-V2 is stop-correct
        out[f"variant_id {v['variant_id'][-2:]}"] = v["variant_id"]
    out = {k: t for k, t in out.items() if len(t) >= 12}
    prov = mech.get("provenance") or {}
    if prov.get("scenario"):  # the scenario name and task id are cues too (Tier 1)
        out["provenance.scenario"] = str(prov["scenario"])
        out["provenance.task_id"] = str(prov["task_id"])
    return out


def _visible(rec: dict) -> str:
    return "\n".join(str(m.get("content", "")) for m in rec.get("messages") or [])


def _records(directory: Path):
    for log in sorted(directory.glob("*.jsonl")):
        for line in log.read_text().splitlines():
            if line.strip():
                yield log.name, json.loads(line)


def blinding_hits(text: str) -> list[str]:
    hits = [name for name in ARM_NAMES if name.lower() in text.lower()]
    hits += [mode for mode in REVIEW_MODES if re.search(rf"(?<![A-Za-z]){re.escape(mode)}(?![A-Za-z])", text)]
    hits += [label for label in CONDITION_LABELS if re.search(rf"\b{label}\b", text)]  # upper-case labels only
    return hits


def episode_map(run: Path) -> dict[str, tuple[str, str]]:
    """episode_id -> (mechanism_id, variant_id) from the run's own assignment table (episode ids are opaque hashes)."""
    path = run / "assignment_table.json"
    out: dict[str, tuple[str, str]] = {}
    if path.is_file():
        for row in json.loads(path.read_text()).get("rows") or []:
            mid, _, vid = str(row.get("instance_id") or "").partition("/")
            if mid and vid:
                out[str(row.get("episode_id"))] = (mid, vid)
    return out


def audit(run: Path) -> dict:
    source = json.loads((bi.TREE / bi.SOURCE_REL).read_text())
    mechs = {m["mechanism_id"]: m for m in source["mechanisms"]}
    # runs made before the suite ids were renamed used AA-<scenario>-<task> ids: keep them auditable through provenance
    for m in list(mechs.values()):
        prov = m.get("provenance") or {}
        if prov.get("scenario"):
            mechs.setdefault(f"AA-{prov['scenario']}-{prov['task_id']}", m)
    episodes = episode_map(run)
    report = {"run": str(run), "calls": 0, "main_calls": 0, "detention_calls": 0, "episodes_with_log": 0, "m1_missing": [],
              "call_errors": [], "refusal_calls": 0, "leaks": [], "m2_calls_checked": 0, "m2_unattributed_calls": 0,
              "m3_calls_checked": 0, "m3_hits": [], "non_completed": [], "result_files": 0}
    for path in sorted((run / "episodes").glob("*.json")):
        report["result_files"] += 1
        data = json.loads(path.read_text())
        if data.get("status") != "completed":
            report["non_completed"].append({"file": path.name, "status": data.get("status"), "error": str(data.get("error"))[:200]})
    for phase, directory in (("main", run / "episode_calls"), ("detention", run / "detention" / "episode_calls")):
        if not directory.is_dir():
            continue
        seen_logs = set()
        for name, rec in _records(directory):
            seen_logs.add(name)
            report["calls"] += 1
            report[f"{phase}_calls"] += 1
            if "messages" not in rec or "reply_text" not in rec:
                report["m1_missing"].append(f"{phase}/{name}")
            kind = rec.get("error_kind") or ""
            if kind == "PROVIDER_REFUSAL":
                report["refusal_calls"] += 1
            elif rec.get("error") or kind:
                report["call_errors"].append({"log": f"{phase}/{name}", "error": str(rec.get("error") or kind)[:200]})
            visible = _visible(rec)
            ep = str(rec.get("episode_id") or "")
            if ep in episodes:
                mid, vid = episodes[ep]
            else:
                match = EPISODE_RE.match(ep)
                mid, vid = (match["mid"], f"{match['mid']}-{match['v']}") if match else ("", "")
            if mid in mechs:
                report["m2_calls_checked"] += 1
                for label, text in hidden_strings(mechs[mid], vid, phase=phase).items():
                    if text in visible:
                        report["leaks"].append({"log": f"{phase}/{name}", "episode": rec.get("episode_id"), "what": label})
            else:
                report["m2_unattributed_calls"] += 1
            if phase == "detention":
                report["m3_calls_checked"] += 1
                hits = blinding_hits(visible)
                if hits:
                    report["m3_hits"].append({"log": f"detention/{name}", "episode": rec.get("episode_id"), "found": hits})
        report["episodes_with_log"] += len(seen_logs) if phase == "main" else 0
    summary_path = run / "detention" / "detention_summary.json"
    if summary_path.is_file():
        d = json.loads(summary_path.read_text())
        report["detention"] = {k: d.get(k) for k in (
            "status", "selected_episodes", "rows_planned", "rows_completed", "rows_refused", "rows_failed", "rows_interrupted",
            "route_refusals", "route_errors", "model_calls_total", "max_model_calls")}
    report["ok_m1"] = not report["m1_missing"] and report["calls"] > 0
    report["ok_m2"] = not report["leaks"] and report["m2_calls_checked"] > 0 and report["m2_unattributed_calls"] == 0
    report["ok_m3"] = report["m3_calls_checked"] > 0 and not report["m3_hits"]
    report["m3_exercised"] = report["m3_calls_checked"] > 0
    report["ok_no_errors"] = not report["call_errors"] and not report["non_completed"]
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    report = audit(Path(args.run))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    skip = ("leaks", "call_errors", "m1_missing", "non_completed", "m3_hits")
    print(json.dumps({k: v for k, v in report.items() if k not in skip}, sort_keys=True))
    for key in skip:
        if report[key]:
            print(key, report[key][:5])
    ok = report["ok_m1"] and report["ok_m2"] and report["ok_no_errors"] and not report["m3_hits"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
