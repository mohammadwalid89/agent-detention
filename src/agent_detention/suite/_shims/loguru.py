"""Minimal stand-in for loguru (not installed here; no network).  Logging is dropped."""
class _Logger:
    def __getattr__(self, name):
        return lambda *a, **k: None
logger = _Logger()
