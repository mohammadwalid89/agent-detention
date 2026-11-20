"""The 10 Tier 2 pairs for gate items 1-2 (rule fixed before any control was run on them).

Strata (count): airline act tasks 3, retail 3, airline native-stop 1, telecom 1, CRM 2.  Inside a stratum the pairs with
the lowest sha256(seed|mechanism_id) are taken.  Writes the selection and subset files for ``suite/tier2/dryrun.py``.
"""

from __future__ import annotations

import hashlib
import json
import sys

from agent_detention import paths

SEED = 20261118
QUOTA = [("T2-AIRLINE-NS-", 1), ("T2-AIRLINE-", 3), ("T2-RETAIL-", 3), ("T2-TELECOM-", 1), ("T2-CRM-", 2)]


def stratum(mid: str) -> str:
    for prefix, _ in QUOTA:  # NS first: it is a prefix-specialisation of T2-AIRLINE-
        if mid.startswith(prefix):
            return prefix
    raise ValueError(mid)


def select(mechanisms: list[dict]) -> list[str]:
    out = []
    for prefix, n in QUOTA:
        pool = sorted((m["mechanism_id"] for m in mechanisms if stratum(m["mechanism_id"]) == prefix),
                      key=lambda mid: hashlib.sha256(f"{SEED}|{mid}".encode()).hexdigest())
        out += pool[:n]
    return out


def main() -> int:
    mechs = json.loads((paths.tier2_out() / "tau2_mechanisms.json").read_text())
    text = json.loads((paths.tier2_out() / "tau2_case_text.json").read_text())
    ids = select(mechs["mechanisms"])
    out = paths.reports() / "tier2_compat10"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ids.json").write_text(json.dumps(ids, indent=1) + "\n")
    sub = {**mechs, "mechanisms": [m for m in mechs["mechanisms"] if m["mechanism_id"] in ids]}
    sub["n"] = len(sub["mechanisms"])
    (out / "mechanisms.json").write_text(json.dumps(sub, sort_keys=True))
    (out / "case_text.json").write_text(json.dumps({**text, "mechanisms": {k: v for k, v in text["mechanisms"].items() if k in ids}}, sort_keys=True))
    print(len(ids), ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
