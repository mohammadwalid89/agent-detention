"""Minimal stand-in for addict.Dict: a dict with recursive update, as tau2's pydantic_utils uses it."""
class Dict(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
    def update(self, other=None, **k):
        other = dict(other or {}, **k)
        for key, val in other.items():
            if isinstance(val, dict) and isinstance(self.get(key), dict):
                cur = Dict(self[key])
                cur.update(val)
                self[key] = cur
            else:
                self[key] = val
