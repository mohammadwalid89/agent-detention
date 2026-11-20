"""Offline dry run of suite mechanisms through the EXISTING runner with the scripted controls.

No model call, no network.  It uses the harness's own episode machinery (``run_episode`` with the real
world, authority checks and verifier) and the five scripted floor policies from
``tools/delegation/floors.py`` (uniform-random, always-act, always-escalate, rule oracle,
elimination retry) plus the pilot's HEURISTIC control, unchanged.  Nothing in the harness tree is
edited: the only adaptation is *in-process*: the case-text overlay lookup is pointed at the suite's
overlay (the tree's loader is hard-wired to the set-12 file).

    AD_TREE=<patched harness tree> python -m agent_detention.suite.tier2.dryrun \
        --mechanisms tau2_mechanisms.json --case-text tau2_case_text.json --out dryrun_tau2 [--seeds 20]

Expected pattern for a well-formed pair set: always-act passes every V1 and fails every V2;
always-escalate the reverse; the rule oracle passes all; uniform-random lands well below 100%.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

from agent_detention.suite.tier1 import harness


def _load_floors():
    sys.dont_write_bytecode = True
    tree = harness.use_tree()
    spec = importlib.util.spec_from_file_location("floors_tool", tree / "tools/delegation/floors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_instances(mech_path: Path, text_path: Path):
    from vera.delegation import case_text as ct
    from vera.pilot.instances import _instance_from_record

    mechs = json.loads(mech_path.read_text())["mechanisms"]
    table = json.loads(text_path.read_text())
    # in-process only: make the harness's own overlay lookup read the suite overlay
    ct.load_case_text = lambda path=None: table  # noqa: E731
    instances = []
    for m in mechs:
        for v in m["variants"]:
            instances.append(_instance_from_record(m, v))
    return instances, mechs


def ordered_controls(floors, ct, instances):
    """Chain-aware versions of always-act and the rule oracle.

    The stock controls assume ONE act edge per case (always-act takes the alphabetically first
    non-stop action; the oracle reads the guards of the edge that completes the case and never
    repeats an action).  Tau2 cases can need 2-3 writes in order (even the same tool twice), with
    the policy gate on the FIRST write.  These two controls walk the act chain in order and read the
    first edge's gate.  They are suite-level additions; the harness's scoring is not touched.
    """
    from vera.delegation.policies import ABSTAIN_ORDER

    def act_path(instance):
        return [e["action"] for e in instance.graph["edges"] if e.get("preferred")]

    def ordered_act(path):
        state = {"i": 0}

        def body(observation):
            last = observation.get("last_result") or {}
            if last.get("decision") == "APPLIED":
                state["i"] += 1
            if state["i"] >= len(path):
                return {"done": True}
            return {"action": path[state["i"]], "reason": "always act, in chain order"}
        return floors._with_report(body)

    def ordered_oracle(instance):
        first = next(e for e in instance.graph["edges"] if e.get("preferred"))
        labels = [instance.signal_labels.get(p, p) for p in first.get("requires", [])]
        path = act_path(instance)
        stop = [str(a) for a in (instance.mechanism.get("abstention_contract") or {}).get("accepted_path") or []]
        state = {"i": 0, "stopped": False}

        def body(observation):
            signals = observation["observations"]["signals"]
            if "act" not in state:  # decide once, on the first observation (the gate flips after step 1)
                state["act"] = all(signals.get(lbl) for lbl in labels)
            act = state["act"]
            last = observation.get("last_result") or {}
            if last.get("decision") == "APPLIED":
                state["i"] += 1
            if act:
                if state["i"] >= len(path):
                    return {"done": True}
                return {"action": path[state["i"]], "reason": "rule holds: act"}
            if not state["stopped"]:
                state["stopped"] = True
                actions = observation["observations"]["permitted_actions"]
                for a in stop + list(ABSTAIN_ORDER):
                    if a in actions:
                        return {"action": a, "reason": "rule violated: stop"}
            return {"done": True}
        return floors._with_report(body)

    out = {}
    for inst in instances:
        labelled = ct.apply_case_text(inst, ct.load_case_text())
        cell = floors.cell_for(inst, "v14")
        a = floors.passed(inst, floors._episode(cell, ordered_act(act_path(inst)), "oact"))
        o = floors.passed(inst, floors._episode(cell, ordered_oracle(labelled), "oor"))
        out[inst.instance_id] = {"ordered_always_act": float(a), "ordered_oracle": float(o)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanisms", required=True)
    ap.add_argument("--case-text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="first N mechanisms only (smoke)")
    args = ap.parse_args(argv)

    floors = _load_floors()
    from vera.delegation import case_text as ct

    instances, mechs = load_instances(Path(args.mechanisms), Path(args.case_text))
    if args.limit:
        keep = {m["mechanism_id"] for m in mechs[: args.limit]}
        instances = [i for i in instances if i.mechanism_id in keep]
    # floors.evaluate labels with the default overlay; point it at the suite overlay as well
    floors.apply_case_text = lambda inst, table=None: ct.apply_case_text(inst, ct.load_case_text())
    raw = floors.evaluate(instances, "v14", args.seeds)
    report = floors.summarise(raw, set_name="suite-tier2", protocol="v14")
    ordered = ordered_controls(floors, ct, instances)
    for row in raw["per_case"]:
        row["scores"].update(ordered[row["instance_id"]])

    # pair-level view: did the controls behave as a sibling set must?
    by_mech = collections.defaultdict(dict)
    for row in raw["per_case"]:
        mid, vid = row["instance_id"].split("/")
        by_mech[mid][vid.rsplit("-", 1)[-1]] = row["scores"]
    problems = []
    pairs = {}
    for mid, v in sorted(by_mech.items()):
        a, b = v.get("V1"), v.get("V2")
        checks = {
            "always_act_wins_A": a["ordered_always_act"] == 1.0,
            "always_act_loses_B": b["ordered_always_act"] == 0.0,
            "always_stop_loses_A": a["always_escalate"] == 0.0,
            "always_stop_wins_B": b["always_escalate"] == 1.0,
            "oracle_wins_both": a["ordered_oracle"] == 1.0 and b["ordered_oracle"] == 1.0,
        }
        pairs[mid] = {"A": a, "B": b, "checks": checks}
        if not all(checks.values()):
            problems.append({"mechanism_id": mid, "failed": [k for k, ok in checks.items() if not ok]})
    n = len(by_mech)
    rate = lambda f: sum(1 for p in pairs.values() if p["checks"][f]) / n  # noqa: E731
    summary = {
        "n_pairs": n, "n_instances": len(instances), "seeds": args.seeds,
        "control_pattern": {k: round(rate(k), 4) for k in
                            ("always_act_wins_A", "always_act_loses_B", "always_stop_loses_A",
                             "always_stop_wins_B", "oracle_wins_both")},
        "overall_stock_floors": report["overall"], "by_label_stock_floors": report["by_label"],
        "overall_ordered": {k: round(sum(r["scores"][k] for r in raw["per_case"]) / len(raw["per_case"]), 4)
                            for k in ("ordered_always_act", "ordered_oracle")},
        "problem_pairs": problems,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dryrun_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    (out / "dryrun_pairs.json").write_text(json.dumps(pairs, indent=1, sort_keys=True) + "\n")
    (out / "dryrun_floors.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("n_pairs", "n_instances", "control_pattern")}, indent=1))
    print("stock floors:", json.dumps(summary["overall_stock_floors"], sort_keys=True))
    print("ordered controls:", json.dumps(summary["overall_ordered"], sort_keys=True))
    print("problem pairs:", len(problems))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
