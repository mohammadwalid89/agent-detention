"""Assignment-row generator for suite pairs (the ``--assignment`` table the pilot scope requires).

    python -m agent_detention.suite.integrate.rows --ids-file ids.json --slot O1 --repeats 1 --out rows.json

One pair = two rows per (slot, repeat): V1 (act-correct) under ``OPTIONAL`` and V2 (stop-correct) under ``ABSTENTION``
(the conditions the v14 manifest admits for them), protocol ``v14``, solo.  Rows are a pure function of the pair list, the
slots, the repeat count and a seed (no outcome-dependent selection); the file carries the digest the runner computes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from agent_detention.suite.integrate import build_inputs as bi

SEED0 = 20261118
# Under the claude_only registry profile the slot names are kept but re-mapped: O1 = Sonnet 5.5 (self), O2 = Haiku 5.5 (self)
# (see identity_assignment.registry_slots_for).  The Sonnet smoke therefore uses O1.  The registry's own `identity_slot: O2`
# field on the Sonnet route belongs to the OTHER (non-pilot) profile and is not the pilot slot name.
DEFAULT_SLOT = "O1"


def opaque_episode_id(slot: str, repeat: int, instance_id: str) -> str:
    return "ep-" + hashlib.sha256(f"{slot}|{repeat}|{instance_id}".encode()).hexdigest()[:12]


def make_rows(mechanism_ids: list[str], slots: list[str], repeats: int, *, source: dict | None = None) -> list[dict]:
    """``source`` = the suite source JSON (to read each mechanism's family and variant ids)."""
    source = source or json.loads((bi.TREE / bi.SOURCE_REL).read_text())
    by_id = {m["mechanism_id"]: m for m in source["mechanisms"]}
    rows: list[dict] = []
    for slot in slots:
        for repeat in range(repeats):
            for index, mid in enumerate(mechanism_ids):
                mech = by_id[mid]
                for variant, condition in zip(mech["variants"], ("OPTIONAL", "ABSTENTION")):
                    instance_id = f"{mid}/{variant['variant_id']}"
                    seed = SEED0 + 1000 * repeat + index
                    rows.append({
                        # OPAQUE on purpose: the harness copies the episode id into the Detention review/retry prompts
                        # (``episode_id`` / ``failed_episode_id``), so an id built from the mechanism and variant ids would show
                        # the model the scenario name, the task id and the V1 (act) / V2 (stop) suffix.
                        "episode_id": opaque_episode_id(slot, repeat, instance_id),
                        "model_slot": slot,
                        "instance_id": instance_id,
                        "family": mech["family"],
                        "condition": condition,
                        "dodge_cue": "",
                        "identity_condition": "NAMED",
                        "repeat": repeat,
                        "seed": seed,
                        "menu_position_seed": seed + 100,
                        "label_permutation": 0,
                        "organization_pattern": "P0_BASE",
                        "protocol": "v14",
                        "solo": True,
                        "resume_key": "|".join([slot, instance_id, condition, "", "NAMED", str(repeat)]),
                    })
    return rows


def write_table(path: str | Path, rows: list[dict], note: str) -> dict:
    table = {
        "kind": "suite.v14b.assignment.v1",
        "formula": "pairs x slots x repeats x {V1:OPTIONAL, V2:ABSTENTION}; protocol v14; solo",
        "expected_episodes": len(rows),
        "note": note,
        "rows": rows,
    }
    Path(path).write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="", help="comma separated mechanism ids")
    ap.add_argument("--ids-file", default="", help="JSON list of mechanism ids")
    ap.add_argument("--slot", action="append", default=[])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    ids = [i for i in args.ids.split(",") if i] or json.loads(Path(args.ids_file).read_text())
    rows = make_rows(ids, args.slot or [DEFAULT_SLOT], args.repeats)
    write_table(args.out, rows, args.note)
    print(f"wrote {args.out}: {len(rows)} rows for {len(ids)} pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
