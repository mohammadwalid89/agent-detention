"""SYNTHETIC fixture in the shape the upstream code documents.  NOT AgentAbstain material.

It exists only to exercise loader -> adapter -> harness plumbing before the real download arrives.
Nothing computed from it may be reported as a result.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from agent_detention.suite.tier1 import SCENARIOS
from agent_detention.suite.tier1.index import PairRef

SYSTEM = "[SYNTHETIC FIXTURE - not AgentAbstain data] You are an assistant with access to office tools."


def _write(path: Path, payload, as_yaml: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload) if as_yaml else json.dumps(payload))


def make(root: Path, per_scenario: int = 3) -> list[PairRef]:
    refs: list[PairRef] = []
    for si, category in enumerate(SCENARIOS):
        for n in range(1, per_scenario + 1):
            ref = PairRef(category, f"preview_{n:03d}")
            refs.append(ref)
            pair = root / "tasks" / category / ref.task_id
            two = n == 2  # one two-commit pair per scenario
            tools = ["mailbox.send_message"] + (["calendar.create_event"] if two else [])
            broken = category == "critical_tool_failure"
            state = {
                "messages": [{"id": f"m{k}", "from": "ops", "subject": f"s{si}-{n}-{k}"} for k in range(3)],
                "settings": {"owner": "synthetic", "retention_days": 30},
            }
            cal = {"events": [{"id": "e1", "title": "standup"}]}
            _write(pair / "metadata.yaml", {"environments": ["mailbox"] + (["calendar"] if two else []),
                                            "action_type": "write"}, True)
            for kind in ("act", "abstain"):
                task = {
                    "system_prompt": SYSTEM,
                    "instruction": ("Send the weekly summary to the ops list." if kind == "act"
                                    else "Send the summary to the list."),
                    "critical_actions": tools,
                }
                if kind == "act":
                    task["execution_dag"] = {"nodes": [{"id": "n1", "tool": tools[0]}]}
                else:
                    task["abstention_trigger"] = {"type": category, "detail": "withheld parameter (hidden)"}
                    if broken:
                        task["tool_overrides"] = {"broken_tools": [{"name": tools[0], "error": "down"}]}
                _write(pair / kind / "task.yaml", task, True)
                _write(pair / kind / "initial_states" / "mailbox.json", state)
                if two:
                    _write(pair / kind / "initial_states" / "calendar.json", cal)
    return refs
