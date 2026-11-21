"""Strict parsing helpers for model output.  No model call lives here."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_json_object(text: str) -> dict:
    """Extract one JSON object from a model reply; raise ValueError otherwise.

    The reply may be wrapped in a fenced block or surrounded by prose, but the object
    itself must parse as JSON.  We never repair a malformed action with a regex guess.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty model reply")
    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)
    candidates.extend(match.group(1).strip() for match in _FENCE_RE.finditer(text))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    # Last attempt: the first balanced object in the text.
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:index + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(value, dict):
                        return value
                    break
    raise ValueError("no JSON object in model reply")


@dataclass(frozen=True)
class ParsedAction:
    ok: bool
    action: str | None
    reason: str
    error: str
    raw: str


def parse_action(text: str, permitted_actions: list[str]) -> ParsedAction:
    try:
        obj = extract_json_object(text)
    except ValueError as exc:
        return ParsedAction(False, None, "", str(exc), text)
    action = obj.get("action", obj.get("tool"))
    reason = obj.get("reason", obj.get("why", ""))
    if not isinstance(action, str) or not action.strip():
        return ParsedAction(False, None, str(reason or ""), "missing string 'action'", text)
    action = action.strip()
    if action not in permitted_actions:
        return ParsedAction(
            False, action, str(reason or ""),
            f"action {action!r} is not in the permitted action list", text,
        )
    return ParsedAction(True, action, str(reason or ""), "", text)
