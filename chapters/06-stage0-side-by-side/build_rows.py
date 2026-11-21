"""Build rows_B1.json and rows_B2.json for Stage 0 with ``agent_detention.suite.integrate.rows``.

    AD_WORK=work python chapters/06-stage0-side-by-side/build_rows.py \\
        --sample results/stage0_sample.json --out-dir stage0

Fixed before any Stage 0 data exist (2026-11-18), procedure:
1. primary = stage0_sample.json primary_pairs (95), sorted; flagged = its 2 flagged pairs.
2. Block 1 = random.Random(20261118).sample(sorted(primary), 20); block 2 = the other 75 primary pairs (sorted).
3. Rows for every pair come from ONE rows.make_rows call over sorted(primary) + flagged, slots O1 (Sonnet) and O2 (Haiku),
   1 repeat, so each pair's per-row seed and opaque episode id do not depend on the block it lands in.
4. Order inside a block: the (pair, version) units are shuffled with random.Random(20261118 + block); each unit is written
   as its O1 row then its O2 row (models interleaved, same pair order for both models).  In block 2 the flagged stratum is
   appended after all primary units (shuffled the same way among themselves), so a call-cap stop hits it first.
   NOTE: GridRunner.run() re-orders with runner_v13.interleaved_schedule (round-robin over (condition, slot) groups,
   file order kept inside each group); the dry run records the order actually executed.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from agent_detention.suite.integrate import rows as R

SEED = 20261118
SLOTS = ["O1", "O2"]  # claude_only profile: O1 = Sonnet 5.5, O2 = Haiku 5.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sample", required=True, help="stage0_sample.json from make_sample.py")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sample = json.loads(Path(args.sample).read_text())
    primary = sorted(sample["primary_pairs"])
    flagged = [f["pair"] for f in sample["flagged_pairs"]]
    assert len(primary) == 95 and len(set(primary)) == 95 and len(flagged) == 2 and not set(flagged) & set(primary)
    b1 = random.Random(SEED).sample(primary, 20)
    b2 = [p for p in primary if p not in set(b1)]
    all_rows = R.make_rows(primary + flagged, SLOTS, 1)
    by_unit: dict[tuple[str, str], dict[str, dict]] = {}
    for row in all_rows:
        mid, _, vid = row["instance_id"].partition("/")
        by_unit.setdefault((mid, vid), {})[row["model_slot"]] = row

    def units(pairs):
        return [u for u in by_unit if u[0] in set(pairs)]

    def ordered(pairs, rng):
        us = sorted(units(pairs))
        rng.shuffle(us)
        return [by_unit[u][s] for u in us for s in SLOTS]

    rows_b1 = ordered(b1, random.Random(SEED + 1))
    rng2 = random.Random(SEED + 2)
    rows_b2 = ordered(b2, rng2) + ordered(flagged, rng2)
    meta = {}
    for name, rows, pairs, fl in (("B1", rows_b1, b1, []), ("B2", rows_b2, b2, flagged)):
        path = out / f"rows_{name}.json"
        R.write_table(path, rows, f"Stage 0 block {name[-1]} (PILOT): {len(pairs)} primary pairs"
                      + (f" + {len(fl)} flagged pairs (own stratum, last)" if fl else "")
                      + " x {V1,V2} x {O1 Sonnet 5.5, O2 Haiku 5.5} x 1 repeat; seed 20261118; procedure in "
                      "ad-stage0-launch-20261118/tools/build_rows.py")
        table = json.loads(path.read_text())
        table["stage0"] = {
            "block": name, "prereg_sha256": "958e2ed5dc3d8e5f587f0e1f9248353df7457045b19a04ebe01c02fdab3f5883",
            "primary_pairs": sorted(pairs), "flagged_pairs": fl,
            "sample_file_sha256": hashlib.sha256(Path(args.sample).read_bytes()).hexdigest(),
        }
        # compact, one row per line, so a diff of two row files is readable
        body = ",\n".join(json.dumps(x, sort_keys=True, separators=(",", ":")) for x in table.pop("rows"))
        head = json.dumps(table, sort_keys=True, separators=(",", ":"))
        path.write_text(head[:-1] + ',\n"rows":[\n' + body + "\n]}\n")
        assert json.loads(path.read_text())["rows"] == rows
        meta[name] = {"rows": len(rows), "pairs": len(pairs) + len(fl),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert meta["B1"]["rows"] == 80 and meta["B2"]["rows"] == 308
    ids = [r["episode_id"] for r in rows_b1 + rows_b2]
    assert len(ids) == len(set(ids)) == 388
    meta["block1_pairs"] = b1
    (out / "rows_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "block1_pairs"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
