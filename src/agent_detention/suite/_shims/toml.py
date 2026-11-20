"""Minimal stand-in for `toml` using the stdlib tomllib (read only)."""
import tomllib


def load(fp, **k):
    data = fp.read()
    if isinstance(data, str):
        data = data.encode()
    return tomllib.loads(data.decode())
def loads(s, **k):
    return tomllib.loads(s)
def dump(*a, **k):
    raise NotImplementedError("shim is read-only")
