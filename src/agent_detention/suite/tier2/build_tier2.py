"""Build the Tier 2 case files (tau2-bench airline, retail, telecom; CRM payments slice).

    TAU2_ROOT=... CRM_BENCH_ROOT=... python -m agent_detention.suite.tier2.build_tier2

Outputs go to ``$AD_WORK/tier2`` (deterministic; no model call, no network):
  tau2_mechanisms.json   harness-format mechanisms (two variants each: V1 act, V2 stop)
  tau2_case_text.json    goal text and signal labels per mechanism (overlay; not owner-signed)
  tau2_exclusions.json   every task that did not become a pair, with the exact reason
  tau2_build_stats.json  counts
  crm_inventory.json     task counts of the CRM benchmark by domain
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

from agent_detention import paths

from . import crm_payments as C
from . import record_builder as B
from . import tau2_adapter as A
from . import tau2_native_stop as N
from . import tau2_replay as R
from . import tau2_telecom as T


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def crm_inventory() -> dict:
    """Task counts of the whole CRM-Agent-Bench catalogue by domain and by native-contract shape."""
    import glob

    import yaml
    out = {"total": 0, "domains": {}}
    for f in sorted(glob.glob(str(C.crm_root() / "src/tasks/*.yaml"))):
        tasks = yaml.safe_load(open(f))["tasks"]
        d = collections.Counter()
        for t in tasks:
            d["tasks"] += 1
            d["no_expected_actions"] += not (t.get("expected_actions") or [])
            d["with_forbidden_actions"] += bool(t.get("forbidden_actions"))
            d["with_expected_state"] += bool(t.get("expected_state"))
            d["with_forbidden_state"] += bool(t.get("forbidden_state"))
            d["with_initial_state"] += bool(t.get("initial_state"))
        out["domains"][Path(f).stem] = dict(d)
        out["total"] += d["tasks"]
    return out


def main() -> int:
    OUT = paths.tier2_out()
    OUT.mkdir(parents=True, exist_ok=True)
    mechs, texts, exclusions = [], {}, []
    stats = collections.OrderedDict()
    for domain in ("airline", "retail"):
        tasks = R.load_tasks(domain)
        base_ids = set(map(str, R.load_split(domain)["base"]))
        c = collections.Counter()
        for task in tasks:
            if str(task["id"]) not in base_ids:
                continue
            c["tasks"] += 1
            res = A.build_pair(domain, task)
            if res["status"] == "built":
                m = res["mechanism"]
                labels = B.check_labels(m)
                ok = (labels[m["variants"][0]["variant_id"]]["label"] == "SOLVABLE"
                      and labels[m["variants"][1]["variant_id"]]["label"] == "ABSTENTION_CORRECT"
                      and m["readiness"]["benchmark_ready"])
                if not ok:
                    exclusions.append({"domain": domain, "task_id": str(task["id"]),
                                       "reason": "harness_labels_wrong", "labels": labels})
                    c["excluded:harness_labels_wrong"] += 1
                    continue
                mechs.append(m)
                texts[m["mechanism_id"]] = res["case_text"]
                c["pairs_native_act__derived_stop"] += 1
            else:
                exclusions.append({"domain": domain, "task_id": res["task_id"], "reason": res["reason"]})
                c["excluded:" + res["reason"].split(":")[0]] += 1
        stats[domain] = dict(c)
    # ---- airline native-stop tasks: native stop + DERIVED act (cancellation requests only)
    nbuilt, nex = N.build_all()
    c = collections.Counter()
    for b in nbuilt:
        m = b["mechanism"]
        labels = B.check_labels(m)
        v1, v2 = (v["variant_id"] for v in m["variants"])
        if (labels[v1]["label"] == "SOLVABLE" and labels[v2]["label"] == "ABSTENTION_CORRECT"
                and m["readiness"]["benchmark_ready"]):
            mechs.append(m)
            texts[m["mechanism_id"]] = b["case_text"]
            c["pairs_native_stop__derived_act"] += 1
            c["tasks_represented"] += 1 + len(m["suite"]["also_represents_task_ids"])
        else:
            exclusions.append({"domain": "airline", "task_id": m["mechanism_id"], "reason": "harness_labels_wrong"})
    exclusions.extend(nex)
    stats["airline_native_stop"] = dict(c)
    # ---- telecom: assistant-side, DB-checkable decisions only (3 pairs from 5 DB structures)
    built, tex, structs = T.build_all()
    c = collections.Counter()
    base = [t for t in T.load_tasks() if t["id"] in T.base_ids()]
    c["tasks"] = len(base)
    c["db_structures"] = len(structs)
    for b in built:
        m = b["mechanism"]
        labels = B.check_labels(m)
        v1, v2 = (v["variant_id"] for v in m["variants"])
        if labels[v1]["label"] == "SOLVABLE" and labels[v2]["label"] == "ABSTENTION_CORRECT" and m["readiness"]["benchmark_ready"]:
            mechs.append(m)
            texts[m["mechanism_id"]] = b["case_text"]
            c["pairs_native_act__derived_stop"] += 1
            c["tasks_represented_by_a_pair"] += len(m["suite"]["represents"])
        else:
            exclusions.append({"domain": "telecom", "task_id": m["suite"]["task_id"], "reason": "harness_labels_wrong"})
    exclusions.extend(tex)
    stats["telecom"] = dict(c)
    # ---- CRM-Agent-Bench: payments-transfer cluster (native <-> native / native + derived act)
    cbuilt, cex = C.build_all()
    c = collections.Counter()
    inv = crm_inventory()
    c["tasks_in_dataset"] = inv["total"]
    for b in cbuilt:
        m = b["mechanism"]
        labels = B.check_labels(m)
        v1, v2 = (v["variant_id"] for v in m["variants"])
        if (labels[v1]["label"] == "SOLVABLE" and labels[v2]["label"] == "ABSTENTION_CORRECT"
                and m["readiness"]["benchmark_ready"]):
            mechs.append(m)
            texts[m["mechanism_id"]] = b["case_text"]
            key = "pairs_native_native" if m["variants"][0]["origin"] == "native" else "pairs_native_stop__derived_act"
            c[key] += 1
        else:
            exclusions.append({"domain": "crm_payments", "task_id": m["mechanism_id"],
                               "reason": "harness_labels_wrong"})
    exclusions.extend(cex)
    stats["crm_agent_bench"] = dict(c)
    dump(OUT / "crm_inventory.json", inv)
    dump(OUT / "tau2_mechanisms.json", {"kind": "suite.tier2.tau2.v1", "n": len(mechs), "mechanisms": mechs})
    dump(OUT / "tau2_case_text.json", {"kind": "suite.tier2.tau2.case_text.v1",
                                       "status": "DRAFT_NOT_OWNER_SIGNED", "mechanisms": texts})
    dump(OUT / "tau2_exclusions.json", exclusions)
    dump(OUT / "tau2_build_stats.json", stats)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
