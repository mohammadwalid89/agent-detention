"""Canonical serialization and hashing.

Rule (design §4): hash-relevant rows contain no floats. Integers and decimal
strings only. A float in a hash-relevant field raises instead of silently
producing an unverifiable hash.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .errors import ContractError

_NO_FLOAT_MSG = (
    "float in hash-relevant field at {path!r}; use int or a decimal string "
    "(design §4: no floats in hash-relevant fields)"
)


def _check_hash_safe(value: Any, path: str = "") -> None:
    if isinstance(value, float):
        raise ContractError(_NO_FLOAT_MSG.format(path=path or "<root>"))
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ContractError(f"non-string key at {path!r}: {k!r}")
            _check_hash_safe(v, f"{path}.{k}" if path else k)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _check_hash_safe(v, f"{path}[{i}]")
    elif isinstance(value, (str, int, bool)) or value is None:
        return
    else:
        raise ContractError(f"non-canonical type at {path!r}: {type(value).__name__}")


def canonical_json(value: Any, *, hash_safe: bool = True) -> str:
    """Deterministic JSON: UTF-8, sorted keys, no insignificant whitespace."""
    if hash_safe:
        _check_hash_safe(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def dstr(x: float, places: int = 6) -> str:
    """Decimal string for a non-hash-relevant measurement that must be stored as text."""
    if not math.isfinite(x):
        raise ContractError(f"non-finite measurement: {x!r}")
    return f"{x:.{places}f}"


def chain_hash(prev_hash: str, row: dict) -> str:
    return sha256_text(prev_hash + canonical_json(row))


class HashChain:
    """In-memory hash chain used by the DB layer and by offline bundle verification."""

    GENESIS = "0" * 64

    def __init__(self) -> None:
        self._head = self.GENESIS
        self._rows: list[tuple[int, str, str]] = []

    @property
    def head(self) -> str:
        return self._head

    def append(self, row: dict) -> str:
        row_hash = chain_hash(self._head, row)
        self._rows.append((len(self._rows), self._head, row_hash))
        self._head = row_hash
        return row_hash

    def verify(self, rows: list[dict]) -> bool:
        head = self.GENESIS
        for row in rows:
            head = chain_hash(head, row)
        return head == self._head
