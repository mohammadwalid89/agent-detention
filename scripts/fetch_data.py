"""Download the public datasets the suite adapts. Nothing is redistributed by this repository.

    python scripts/fetch_data.py --dest data [--only agentabstain-data]

Each source is cloned at the commit the reported results were built from. Licences:

    AgentAbstain data   CC-BY-4.0    credit the authors, link the licence, say what was changed
    AgentAbstain code   MIT          only the pair index is read
    tau2-bench          MIT          task data, domain databases, policy text and tool code (imported, not modified)
    crm-agent-bench     Apache-2.0   task catalogue, payments policy and tool environment (imported, not modified)

After the download, export the variables the code reads:

    export AGENTABSTAIN_DATA=data/agentabstain-data AGENTABSTAIN_REPO=data/agentabstain
    export TAU2_ROOT=data/tau2-bench CRM_BENCH_ROOT=data/crm-agent-bench
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

SOURCES = {
    "agentabstain-data": ("https://huggingface.co/datasets/antiquality/agentabstain",
                          "842228426c2a703347396501af61c7890972c7ee", "CC-BY-4.0"),
    "agentabstain": ("https://github.com/AntiQuality/agentabstain",
                     "cfc3faf7ab1cfd4892cde1158d6e43b2f312ddc3", "MIT"),
    "tau2-bench": ("https://github.com/sierra-research/tau2-bench",
                   "c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4", "MIT"),
    "crm-agent-bench": ("https://github.com/salesforce-misc/crm-agent-bench",
                        "fc9575cf9bda085aa1f66d920fea944588005964", "Apache-2.0"),
}


def fetch(name: str, dest: Path) -> None:
    url, commit, licence = SOURCES[name]
    target = dest / name
    if target.exists():
        print(f"{name}: {target} exists, skipped")
        return
    subprocess.run(["git", "clone", "--quiet", url, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--quiet", commit], check=True)
    print(f"{name}: {url} @ {commit[:12]} ({licence})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data")
    ap.add_argument("--only", choices=sorted(SOURCES), action="append")
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in args.only or SOURCES:
        fetch(name, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
