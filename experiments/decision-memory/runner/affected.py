#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The operate loop: a decision changed, which tasks must be retested?

A task is affected when the changed record's scope covers a file the task
touches, or when the task names the record (or the one it supersedes).

    python -m runner.affected DR-0011                 # list
    python -m runner.affected DR-0011 --run --client responses:http://127.0.0.1:8080 --out runs/retest

`--run` hands the affected task ids to run_eval with the remaining
arguments. Wire this to CI on `decisions/**`: the merge that accepts a
record is the trigger.
"""

from __future__ import annotations

import subprocess
import sys

from .tasks import FIXTURE_REPO, load_tasks

sys.path.insert(0, str(FIXTURE_REPO))
from ledgerline.decisions import load_records  # noqa: E402


def affected(record_ids: list[str]) -> list[str]:
    records = {r.id: r for r in load_records(FIXTURE_REPO / "decisions")}
    out = []
    for t in load_tasks():
        hit = False
        for rid in record_ids:
            r = records[rid]
            lineage = {rid, r.supersedes, r.superseded_by} - {None}
            if lineage & set(t.decisions) or r.matches(t.files):
                hit = True
        if hit:
            out.append(t.id)
    return out


def main(argv: list[str]) -> int:
    ids = [a for a in argv if a.startswith("DR-")]
    rest = [a for a in argv if not a.startswith("DR-") and a != "--run"]
    if not ids:
        print(__doc__)
        return 2
    tasks = affected(ids)
    print("affected tasks:", ", ".join(tasks) or "none")
    if "--run" in argv and tasks:
        cmd = [sys.executable, "-m", "runner.run_eval", "--tasks", ",".join(tasks)] + rest
        print("running:", " ".join(cmd))
        return subprocess.call(cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
