"""Solvability scan for the Tier 1 (AgentAbstain) act versions. Deterministic; no model calls; read-only on its inputs.

Question: for each of the 131 adapted Tier 1 act versions, does the task require a tool or an
action that is not in that case's permitted actions?

Inputs (read-only):
  * the adapted record set the runner is built from: ``$AD_WORK/tier1/tier1_set.json`` (``build_tier1_records.py``);
  * the AgentAbstain snapshot ``$AGENTABSTAIN_DATA`` (task.yaml of each act variant, and each environment's
    ``tool_kinds`` table, read with ``ast``; dataset code is never imported or executed).

Checks on the act version (V1) -- each fires a reason code:
  S1_REQUIRED_STATE_UNREACHABLE  (unsolvable) the required postconditions cannot be reached from the entry node using
                                 permitted, non-forbidden actions whose predicates hold in V1, within MAX_STEPS moves.
  S2_MANDATED_TOOL_NOT_PERMITTED (unsolvable) a sentence of the task text names a catalog tool that is not a permitted
                                 action AND contains mandate wording (MUST / must call|use / do not skip / do not
                                 substitute ...). The text forbids the only substitute the projection offers (tables).
  S3_REFERENCE_COMMIT_NOT_PERMITTED (unsolvable) the upstream reference solution (execution_dag) contains a commit-kind
                                 step whose tool is not a permitted action.
  U1_NAMED_TOOL_NOT_PERMITTED    (unclear) the task text names a non-permitted catalog tool as the means, without mandate
                                 wording. The projection shows state as tables instead; whether an agent accepts that
                                 substitute is a judgement call.
  U2_VERIFY_NUMBER_NOT_SHOWN     (unclear) a verify-kind reference step that is not permitted yields a number that does
                                 not appear in the tables or the goal text shown to the agent and is not an equal share
                                 (a shown number / 2..6): the agent must obtain it, and the scan cannot show it can.
                                 (v2: v1 searched the tables only, which flagged numbers stated in the request itself.)
Classification: UNSOLVABLE if any S* fires; else UNCLEAR if any U* fires; else SOLVABLE.

Pair-level check (affects eligibility, reported separately from the act verdict):
  P1_VERSIONS_INDISTINGUISHABLE  V1 and V2 give the agent the same first observation: identical goal text, entry tables
                                 and active predicates (permitted actions are per pair, so identical too). No policy can
                                 tell the versions apart before acting; V2 can only be "held" after the world refuses
                                 the forbidden commit (which the current grading credits as a hold).

Eligible pair = act version SOLVABLE and not P1.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import os
import re
from pathlib import Path

import yaml

from agent_detention import paths

SCAN_VERSION = "a2-solvability-v2"
DATA = Path(os.environ.get("AGENTABSTAIN_DATA", "agentabstain-data"))
SET = paths.tier1_records()

# The runner shows "max_steps": 6 and adds the harness action "read_state".
MAX_STEPS = 6
HARNESS_ACTIONS = frozenset({"read_state"})

MANDATE = re.compile(
    r"\bMUST\b"  # upper case only: lower-case "must" is used for ordinary constraints ("it must be brand new")
    r"|(?i:\bmust\s+(?:call|use|run|invoke)\b)"
    r"|(?i:\bdo\s+not\s+(?:skip|substitute)\b)"
    r"|(?i:\bsubstitute\s+a\s+different\b)"
    r"|(?i:\brequired\s+to\s+(?:call|use)\b)"
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

UNSOLVABLE_CODES = ("S1_REQUIRED_STATE_UNREACHABLE", "S2_MANDATED_TOOL_NOT_PERMITTED", "S3_REFERENCE_COMMIT_NOT_PERMITTED")
UNCLEAR_CODES = ("U1_NAMED_TOOL_NOT_PERMITTED", "U2_VERIFY_NUMBER_NOT_SHOWN")


# ----------------------------------------------------------------------------------------------- inputs

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env_tool_kinds(env: str, data: Path = DATA) -> dict[str, str]:
    """``tool_kinds`` of one environment, by parsing (not importing) its environment.py."""
    tree = ast.parse((data / "environments" / env / "environment.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "tool_kinds" for t in node.targets):
            return dict(ast.literal_eval(node.value))
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "tool_kinds":
            return dict(ast.literal_eval(node.value))
    raise ValueError(f"{env}: no tool_kinds table in environment.py")


def catalog_for(envs: list[str], data: Path = DATA) -> dict[str, str]:
    """Namespaced tool name -> kind, over the pair's environments."""
    out: dict[str, str] = {}
    for env in envs:
        out.update({f"{env}.{name}": kind for name, kind in env_tool_kinds(env, data).items()})
    return out


def act_task_path(record: dict, data: Path = DATA) -> Path:
    p = record["provenance"]
    return data / "tasks" / p["scenario"] / p["task_id"] / "act" / "task.yaml"


def load_act_task(record: dict, data: Path = DATA) -> tuple[dict, bool]:
    """The act task.yaml, and whether its sha256 equals the one the adapter stamped into the record."""
    path = act_task_path(record, data)
    rel = str(path.relative_to(data))
    want = record["provenance"]["source_files_sha256"].get(rel)
    return yaml.safe_load(path.read_text()), (want is not None and want == sha256_file(path))


# ----------------------------------------------------------------------------------------------- checks

def _leaves(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _leaves(v)
    else:
        yield obj


def min_steps_to_goal(record: dict, variant: dict, permitted: set[str], max_steps: int = MAX_STEPS) -> int | None:
    """BFS over the record's state graph from its entry node; None if the required postconditions are unreachable."""
    stg = record["stg"]
    need = set(record["required_postconditions"])
    nodes = {n["id"]: n for n in stg["nodes"]}
    preds = set(variant.get("active_predicates", []))
    forbidden = set(variant.get("forbidden_actions", []))
    out_edges = collections.defaultdict(list)
    for e in stg["edges"]:
        out_edges[e["from"]].append(e)
    frontier, seen, depth = [stg["entry"]], {stg["entry"]}, 0
    while frontier and depth <= max_steps:
        for nid in frontier:
            if need and need <= set(nodes[nid]["postconditions_met"]):
                return depth
        nxt = []
        for nid in frontier:
            for e in out_edges[nid]:
                if e["action"] in permitted and e["action"] not in forbidden and set(e.get("requires", [])) <= preds \
                        and e["to"] not in seen:
                    seen.add(e["to"])
                    nxt.append(e["to"])
        frontier, depth = nxt, depth + 1
    return None


def named_tools(text: str, catalog: dict[str, str]) -> list[tuple[str, str]]:
    """(sentence, namespaced tool) for every catalog tool named in the text, by full or short name, on token boundaries."""
    hits = []
    sentences = [s for s in SENTENCE_SPLIT.split(text) if s.strip()]
    for full in sorted(catalog):
        short = full.split(".", 1)[1]
        pat = re.compile(r"(?<![A-Za-z0-9_])(?:" + re.escape(full) + "|" + re.escape(short) + r")(?![A-Za-z0-9_])")
        for s in sentences:
            if pat.search(s):
                hits.append((s.strip(), full))
    return hits


def _number_shown(x, blob: str) -> bool:
    forms = {repr(x), str(x)}
    if isinstance(x, float) and x.is_integer():
        forms.add(str(int(x)))
        forms.add(f"{x:.2f}")
    return any(re.search(r"(?<![\d.])" + re.escape(f) + r"(?![\d])", blob) for f in forms)


NUM_IN_BLOB = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])")


def _equal_share_derivable(x, blob: str, max_parts: int = 6) -> bool:
    """True if x equals (to the cent) some number shown in the blob divided by 2..max_parts: arithmetic the agent can do."""
    if not isinstance(x, float):  # ints are counts/ids, not shares; integer-valued floats (42.0) are allowed
        return False
    for m in NUM_IN_BLOB.finditer(blob):
        v = float(m.group(1))
        if any(abs(round(v / k, 2) - x) < 0.005 for k in range(2, max_parts + 1)):
            return True
    return False


def classify_act(record: dict, act_task: dict, catalog: dict[str, str]) -> dict:
    """Verdict for the act version (V1) of one adapted record. Pure function of its arguments."""
    v1 = next(v for v in record["variants"] if v["variant_id"].endswith("-V1"))
    permitted = set(record["permitted_actions"]) | HARNESS_ACTIONS
    reasons: list[dict] = []

    steps = min_steps_to_goal(record, v1, permitted)
    if steps is None:
        reasons.append({"code": "S1_REQUIRED_STATE_UNREACHABLE",
                        "detail": f"required postconditions {record['required_postconditions']} unreachable within "
                                  f"{MAX_STEPS} permitted moves"})

    text = f"{act_task['system_prompt']}\n{act_task['instruction']}"
    seen_named = set()
    for sentence, tool in named_tools(text, catalog):
        if tool in permitted or (tool, sentence) in seen_named:
            continue
        seen_named.add((tool, sentence))
        mandated = bool(MANDATE.search(sentence))
        reasons.append({
            "code": "S2_MANDATED_TOOL_NOT_PERMITTED" if mandated else "U1_NAMED_TOOL_NOT_PERMITTED",
            "tool": tool, "tool_kind": catalog[tool], "sentence": sentence[:400],
        })

    dag = (act_task.get("execution_dag") or {}).get("nodes", [])
    # what the agent is shown: the entry tables AND the goal text (system prompt + request)
    blob = json.dumps(v1["entry_tables"], sort_keys=True) + "\n" + str(v1.get("goal_text") or text)
    substituted = collections.Counter()
    derivable_by_division = []
    for n in dag:
        tool, kind = n["tool"], n.get("kind")
        if tool in permitted:
            continue
        substituted[kind] += 1
        if kind == "commit":
            reasons.append({"code": "S3_REFERENCE_COMMIT_NOT_PERMITTED", "tool": tool, "node": n.get("id")})
        if kind == "verify":
            nums = [x for x in _leaves(n.get("must_yield", {})) if isinstance(x, (int, float)) and not isinstance(x, bool)]
            unseen = [x for x in nums if not _number_shown(x, blob)]
            derivable_by_division += [x for x in unseen if _equal_share_derivable(x, blob)]
            unseen = [x for x in unseen if not _equal_share_derivable(x, blob)]
            if unseen:
                reasons.append({"code": "U2_VERIFY_NUMBER_NOT_SHOWN", "tool": tool, "node": n.get("id"),
                                "numbers": unseen})

    codes = sorted({r["code"] for r in reasons})
    verdict = "UNSOLVABLE" if any(c in UNSOLVABLE_CODES for c in codes) else \
        "UNCLEAR" if any(c in UNCLEAR_CODES for c in codes) else "SOLVABLE"
    return {
        "verdict": verdict,
        "codes": codes,
        "reasons": reasons,
        "min_steps_to_goal": steps,
        "permitted_world_actions": sorted(set(record["permitted_actions"])),
        "reference_steps_not_permitted_by_kind": dict(sorted(substituted.items())),
        # descriptive, not a verdict: the agent must do these verify steps by reading the tables
        "verify_steps_replaced_by_tables": substituted.get("verify", 0),
        "verify_numbers_derivable_by_division": derivable_by_division,
    }


OBS_FIELDS = ("goal_text", "entry_tables", "active_predicates")


def pair_indistinguishable(record: dict) -> dict:
    v1 = next(v for v in record["variants"] if v["variant_id"].endswith("-V1"))
    v2 = next(v for v in record["variants"] if v["variant_id"].endswith("-V2"))
    differs = [f for f in OBS_FIELDS if v1.get(f) != v2.get(f)]
    return {"indistinguishable": not differs, "observation_fields_that_differ": differs,
            "hidden_fields_that_differ": sorted(k for k in v1 if k not in OBS_FIELDS and v1[k] != v2.get(k))}


# ----------------------------------------------------------------------------------------------- driver

def scan(set_path: Path = SET, data: Path = DATA) -> dict:
    records = json.load(open(set_path))["mechanisms"]
    cases = []
    for rec in sorted(records, key=lambda r: r["mechanism_id"]):
        act, hash_ok = load_act_task(rec, data)
        cat = catalog_for(rec["provenance"]["environments"], data)
        verdict = classify_act(rec, act, cat)
        pair = pair_indistinguishable(rec)
        mid = rec["mechanism_id"]
        cases.append({
            "mechanism_id": mid,
            "scenario": rec["provenance"]["scenario"],
            "task_id": rec["provenance"]["task_id"],
            "act_task_sha256_matches_record": hash_ok,
            **verdict,
            "pair": pair,
            "eligible": verdict["verdict"] == "SOLVABLE" and not pair["indistinguishable"],
        })
    return {"scan_version": SCAN_VERSION, "inputs": inputs_block(set_path, data), "summary": summarise(cases),
            "cases": cases}


def inputs_block(set_path: Path, data: Path) -> dict:
    return {
        "record_set": str(set_path), "record_set_sha256": sha256_file(set_path),
        "dataset_root": str(data), "max_steps": MAX_STEPS, "harness_actions": sorted(HARNESS_ACTIONS),
        "mandate_regex": MANDATE.pattern,
    }


def summarise(cases: list[dict]) -> dict:
    verdicts = collections.Counter(c["verdict"] for c in cases)
    codes = collections.Counter(code for c in cases for code in c["codes"])
    by_scen: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for c in cases:
        s = by_scen[c["scenario"]]
        s["pairs"] += 1
        s[c["verdict"]] += 1
        s["P1_indistinguishable"] += c["pair"]["indistinguishable"]
        s["eligible"] += c["eligible"]
    return {
        "act_versions": len(cases),
        "verdicts": {k: verdicts.get(k, 0) for k in ("SOLVABLE", "UNSOLVABLE", "UNCLEAR")},
        "reason_code_counts": dict(sorted(codes.items())),
        "pairs_indistinguishable_P1": sum(c["pair"]["indistinguishable"] for c in cases),
        "eligible_pairs": sum(c["eligible"] for c in cases),
        "eligible_if_unclear_included": sum(c["verdict"] != "UNSOLVABLE" and not c["pair"]["indistinguishable"]
                                            for c in cases),
        "act_task_hash_mismatches": sum(not c["act_task_sha256_matches_record"] for c in cases),
        "act_versions_with_a_verify_step_replaced_by_tables": sum(c["verify_steps_replaced_by_tables"] > 0 for c in cases),
        "by_scenario": {k: dict(v) for k, v in sorted(by_scen.items())},
    }


def render_table(result: dict) -> str:
    lines = ["| pair | act verdict | reason codes | P1 | eligible | detail |", "|---|---|---|---|---|---|"]
    for c in result["cases"]:
        detail = "; ".join(
            (f"{r['code'].split('_')[0]}: `{r.get('tool', '').split('.')[-1]}`" +
             (f" ({r['tool_kind']})" if r.get("tool_kind") else "") +
             (f" numbers {r['numbers']}" if r.get("numbers") else ""))
            for r in c["reasons"]) or "-"
        lines.append(f"| {c['mechanism_id'][3:]} | {c['verdict']} | {', '.join(c['codes']) or '-'} | "
                     f"{'yes' if c['pair']['indistinguishable'] else 'no'} | {'yes' if c['eligible'] else 'no'} | "
                     f"{detail} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="solvability.json")
    ap.add_argument("--table", default="solvability_table.md")
    args = ap.parse_args()
    result = scan()
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    Path(args.table).write_text(render_table(result))
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
