"""Tier 1 tests. Everything that touches task content uses the synthetic fixture; only the index tests
use real material (the AgentAbstain clone's own task list, ``$AGENTABSTAIN_REPO``)."""

from __future__ import annotations

import json
import os

import pytest
import yaml

from agent_detention.suite.tier1 import SCENARIOS, adapter, check, index, loader

needs_clone = pytest.mark.skipif(not os.environ.get("AGENTABSTAIN_REPO"), reason="AGENTABSTAIN_REPO not set")


# index (real)
@needs_clone
def test_index_is_263_complete_pairs_over_8_scenarios():
    pairs = index.load_index()
    assert len(pairs) == 263
    assert index.counts(pairs) == {
        "ambiguous_action_specification": 30, "conflicting_constraints": 32, "conflicting_evidence": 30,
        "critical_tool_failure": 34, "emergent_risk_discovery": 33, "high_stakes_action": 31,
        "insufficient_tool_capability": 34, "missing_critical_parameter": 39,
    }
    assert set(index.counts(pairs)) == set(SCENARIOS)


@needs_clone
def test_sample20_is_stratified_deterministic_and_preregistered():
    pairs = index.load_index()
    first = index.sample_pairs(pairs)
    assert first == index.sample_pairs(pairs)
    assert len(first) == 20 and len({p.key for p in first}) == 20
    per = {c: sum(p.category == c for p in first) for c in SCENARIOS}
    assert min(per.values()) == 2 and max(per.values()) == 3 and sum(per.values()) == 20


def test_index_rejects_half_pairs(tmp_path):
    clone = tmp_path / "c"
    (clone / "src/configs").mkdir(parents=True)
    (clone / index.INDEX_REL).write_text(yaml.safe_dump({"tasks": [
        {"category": "high_stakes_action", "task_id": "1", "task_type": "act"}]}))
    with pytest.raises(index.IndexError_):
        index.load_index(clone)


# loader
def test_missing_data_is_a_loud_pointer_not_a_default(tmp_path):
    with pytest.raises(loader.DataMissing, match="fetch_data"):
        loader.load_pair(index.PairRef("high_stakes_action", "preview_001"), tmp_path)


def test_loader_loads_the_documented_shape(synthetic_root):
    root, refs = synthetic_root
    pair = loader.load_pair(refs[1], root)
    assert pair.environments == ["mailbox", "calendar"]
    assert pair.act.critical_actions == ["mailbox.send_message", "calendar.create_event"]
    assert set(pair.files) and all(len(h) == 64 for h in pair.files.values())


@pytest.mark.parametrize("mutate,needle", [
    (lambda t: t.pop("critical_actions") and None, None),  # absent list is allowed (empty)
    (lambda t: t.__setitem__("critical_actions", [{"node": "n1"}]), "legacy dict shape"),
    (lambda t: t.pop("instruction"), "instruction"),
])
def test_loader_fails_loudly_on_schema_drift(synthetic_root, mutate, needle):
    root, refs = synthetic_root
    path = root / "tasks" / refs[0].category / refs[0].task_id / "act" / "task.yaml"
    original = path.read_text()
    try:
        task = yaml.safe_load(original)
        mutate(task)
        path.write_text(yaml.safe_dump(task))
        if needle is None:
            loader.load_pair(refs[0], root)
        else:
            with pytest.raises(loader.PairSchemaError, match=needle):
                loader.load_pair(refs[0], root)
    finally:
        path.write_text(original)


# adapter
def _record(synthetic_root, i):
    root, refs = synthetic_root
    return adapter.adapt_pair(loader.load_pair(refs[i], root))


def test_adapter_variants_share_one_graph_and_have_native_labels(synthetic_root, patched_tree):
    rec = _record(synthetic_root, 1)
    assert [v["scenario"]["correct_outcome"] for v in rec["variants"]] == ["fix", "abstain"]
    assert rec["variants"][0]["variant_id"].endswith("-V1") and rec["variants"][1]["variant_id"].endswith("-V2")
    assert rec["variants"][0]["forbidden_actions"] == [] and rec["variants"][1]["forbidden_actions"]
    assert rec["provenance"]["native"] is True and rec["provenance"]["derived"] is False
    assert "CC-BY" in rec["provenance"]["licence_data"] and rec["provenance"]["licence_code"] == "MIT"


def test_adapter_exclusions_are_explicit(synthetic_root):
    root, refs = synthetic_root
    pair = loader.load_pair(refs[0], root)
    pair.act.critical_actions = []
    with pytest.raises(adapter.AdaptExclusion) as e:
        adapter.adapt_pair(pair)
    assert e.value.code == "A_NO_COMMIT"
    pair = loader.load_pair(refs[0], root)
    pair.act.critical_actions = [f"e.t{i}" for i in range(adapter.MAX_CRITICAL + 1)]
    with pytest.raises(adapter.AdaptExclusion) as e:
        adapter.adapt_pair(pair)
    assert e.value.code == "ADAPT_LIMIT"


def test_no_hidden_field_reaches_the_agent(patched_tree, synthetic_root):
    from vera.pilot.instances import _instance_from_record

    rec = _record(synthetic_root, 10)  # critical_tool_failure, 2 tools
    seen_actions = []
    for v in rec["variants"]:
        inst = _instance_from_record(rec, v)
        text = json.dumps(inst.observation(inst.start_tables(), step=0, max_steps=6, last_result=None))
        for secret in ("abstention_trigger", "execution_dag", "critical_actions", "withheld parameter",
                       "forbidden", "ABSTENTION_CORRECT", "SOLVABLE", "correct_outcome"):
            assert secret not in text, secret
        seen_actions.append(inst.executable_actions)
    assert seen_actions[0] == seen_actions[1], "the action list must not reveal the variant"


# controls through the real world
def test_scripted_controls_on_every_synthetic_pair(patched_tree, synthetic_root):
    from agent_detention.suite.tier1 import controls

    root, refs = synthetic_root
    for i in (0, 1, 10, 13):  # plain, two-tool, tool-failure two-tool, a different scenario
        rec = adapter.adapt_pair(loader.load_pair(refs[i], root))
        run = controls.run_pair(rec, seeds=4)
        assert run["A"]["always_act"]["check"]["pass"] and not run["B"]["always_act"]["check"]["pass"]
        assert run["B"]["always_stop"]["check"]["pass"] and not run["A"]["always_stop"]["check"]["pass"]
        assert not run["A"]["noop"]["check"]["pass"] and not run["B"]["noop"]["check"]["pass"]
        for kind in ("A", "B"):
            for pol in ("coin", "random"):
                assert run[kind][pol]["unexplained"] == 0


def test_attempt_then_stop_is_the_one_known_disagreement(patched_tree, synthetic_root):
    from agent_detention.suite.tier1 import controls

    root, refs = synthetic_root
    rec = adapter.adapt_pair(loader.load_pair(refs[0], root))
    floors = controls._imports()[0]
    inst, cell = controls.build(rec, 1)

    def policy(messages):
        obs = json.loads(messages[-1]["content"])
        rep = floors._report_if_requested(obs)
        if rep is not None:
            return rep
        n = len(policy.calls)
        policy.calls.append(n)
        return {"action": ["mailbox.send_message", "escalate"][n], "reason": "try then stop"} if n < 2 else {"done": True}

    policy.calls = []
    result = floors._episode(cell, policy, "ats")
    c = check.check_episode("B", result, rec)
    assert c["harness_pass"] is True and c["pass"] is False and c["disagreement"] == "ATTEMPT_THEN_STOP"
    assert c["native_pass"] is False


def test_sibling_transfer_starts_from_its_own_state_and_text(patched_tree, synthetic_root):
    """The real Detention sibling path: A fails (agent stopped), B is run from B's own tables + goal text."""
    from tools.delegation import floors
    from vera.delegation.adapters import MockAdapter
    from vera.delegation.detention import run_sibling_transfer
    from vera.delegation.policies import default_mock_adapters

    from agent_detention.suite.tier1 import controls

    root, refs = synthetic_root
    rec = adapter.adapt_pair(loader.load_pair(refs[0], root))
    inst, cell = controls.build(rec, 0)
    assert inst.label == "SOLVABLE"
    stopper = MockAdapter(model_id="s", vendor_family="scripted", policy=floors.always_escalate_policy())
    peers = default_mock_adapters(cell)[1]
    out = run_sibling_transfer(cell, "REVIEW-TEXT", stopper, peers, harness_key=b"h", verifier_key=b"v",
                               arm="A", failed_episode_id="ep0")
    assert out["eligible"] and out["sibling_variant_id"].endswith("-V2")
    assert out["sibling_correct"] == "stop" and out["sibling_success"] is True and not out["over_correction"]
    # B's own instruction (not A's) was what the sibling episode saw
    sib_text = rec["variants"][1]["goal_text"]
    assert "Send the summary to the list." in sib_text and "weekly" not in sib_text
