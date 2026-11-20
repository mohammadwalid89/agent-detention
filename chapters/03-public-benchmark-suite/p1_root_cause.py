"""Read-only root cause for P1 (versions indistinguishable after projection), over all 131 adapted pairs.

Upstream each AgentAbstain pair changes ONE element between act and abstain (``metadata.yaml: transformation_dimension``).
For each pair this script records: the dimension; whether instruction/system prompt and initial states are byte-equal
upstream; the tools the abstain version loses (``abstention_trigger.remaining_tools`` vs the act DAG's tools) or has
broken (``tool_overrides.broken_tools``); and whether any of those tools is a commit tool in the projection (the only
tools the projection shows). Output: JSON on stdout."""
import collections
import json
import os
import sys
from pathlib import Path

import yaml

from agent_detention import paths

DATA = Path(os.environ.get("AGENTABSTAIN_DATA", "agentabstain-data"))
SET = paths.tier1_records()


def states(d: Path) -> dict:
    return {p.name: p.read_bytes() for p in sorted((d / "initial_states").glob("*.json"))}


def main() -> int:
    recs = {r["mechanism_id"]: r for r in json.load(open(SET))["mechanisms"]}
    scan = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else None
    p1 = {c["mechanism_id"]: c["pair"]["indistinguishable"] for c in scan["cases"]} if scan else {}
    rows, tally = [], collections.Counter()
    for mid, r in sorted(recs.items()):
        p = r["provenance"]
        base = DATA / "tasks" / p["scenario"] / p["task_id"]
        meta = yaml.safe_load((base / "metadata.yaml").read_text())
        act = yaml.safe_load((base / "act" / "task.yaml").read_text())
        ab = yaml.safe_load((base / "abstain" / "task.yaml").read_text())
        same_text = (act["system_prompt"], act["instruction"]) == (ab["system_prompt"], ab["instruction"])
        same_state = states(base / "act") == states(base / "abstain")
        dag_tools = {n["tool"] for n in (act.get("execution_dag") or {}).get("nodes", [])}
        remaining = (ab.get("abstention_trigger") or {}).get("remaining_tools")
        removed = sorted(dag_tools - set(remaining)) if remaining is not None else []
        broken = [str(b.get("name")) for b in ((ab.get("tool_overrides") or {}).get("broken_tools") or [])]
        commit = set(r["permitted_actions"]) - {"escalate", "abstain"}
        trigger_tools = set(removed) | set(broken)
        row = {
            "pair": mid, "scenario": p["scenario"], "dimension": meta.get("transformation_dimension"),
            "upstream_same_text": same_text, "upstream_same_state": same_state,
            "tools_removed_in_abstain": removed, "tools_broken_in_abstain": broken,
            "trigger_tool_is_commit_tool": bool(trigger_tools & commit),
            "P1_in_scan": p1.get(mid),
        }
        # the projection can only show the trigger through text, state, or a gated COMMIT tool
        row["projection_carries_trigger"] = (not same_text) or (not same_state) or row["trigger_tool_is_commit_tool"]
        rows.append(row)
        tally[(p["scenario"], row["dimension"], row["projection_carries_trigger"], row["P1_in_scan"])] += 1
    agree = sum(r["projection_carries_trigger"] == (not r["P1_in_scan"]) for r in rows) if scan else None
    print(json.dumps({
        "pairs": len(rows),
        "agreement_with_scan_P1": agree,
        "tally_scenario_dimension_carried_P1": [
            {"scenario": a, "dimension": b, "projection_carries_trigger": c, "P1_in_scan": d, "n": n}
            for (a, b, c, d), n in sorted(tally.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))],
        "rows": rows,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
