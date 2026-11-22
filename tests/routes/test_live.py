"""One real call to the Anthropic Messages API through ``AnthropicApiAdapter``.  Opt-in and paid.

Skipped unless ``ANTHROPIC_API_KEY`` is set and ``AD_LIVE_SMOKE=1``: the second variable keeps a key that happens to
be in the environment from making a paid call on every test run.  The call asks for at most 16 output tokens from the
cheaper pilot route and checks that the provider served the model that was asked for.

    AD_LIVE_SMOKE=1 ANTHROPIC_API_KEY=... python -m pytest tests/routes/test_live.py
"""

from __future__ import annotations

import os

import pytest

from agent_detention.suite.tier1 import harness

harness.use_tree()

from vera.delegation.adapters import SERVED_MODEL_MATCH, AnthropicApiAdapter, served_model_check  # noqa: E402
from vera.delegation.routes.registry import load_registry  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("AD_LIVE_SMOKE") == "1"),
    reason="live call: set ANTHROPIC_API_KEY and AD_LIVE_SMOKE=1",
)


def test_one_live_call_serves_the_requested_model():
    routes = load_registry(profile="claude_only")["routes"]
    model_id = next(r["model_id"] for r in routes if r["route_id"] == "anthropic_claude_haiku_5_5")
    adapter = AnthropicApiAdapter(model_id, max_output_tokens=16, max_retries=0)
    result = adapter.complete([{"role": "user", "content": "Reply with the word OK."}])
    assert result.ok, (result.error_kind, result.error)
    assert result.text.strip()
    assert served_model_check(model_id, result.served_model_id) == SERVED_MODEL_MATCH
    assert adapter.calls == 1
