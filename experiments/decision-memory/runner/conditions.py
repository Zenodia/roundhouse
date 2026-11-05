# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The three evaluation conditions of the memory team's methodology, as
"what decision context reaches the model":

    no_memory   nothing from decisions/
    oracle      every accepted record, in full (the ceiling)
    memory      the records a retrieval rule selects for this task

The retrieval rule in `memory` is deliberately simple and can miss: a
record is selected when its `scope` globs cover a file the task touches.
A task whose decision lives in a file it does not touch (T10 reads
retention from reports.py) is a retrieval miss by construction, and the
gap between `memory` and `oracle` on such tasks is the thing to measure.

`--records all` adds superseded, proposed and rejected records to the
context too (the confusable stressor): the model must then read `status`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .tasks import FIXTURE_REPO, Task

sys.path.insert(0, str(FIXTURE_REPO))
from ledgerline.decisions import Record, current, for_files, load_records  # noqa: E402

CONDITIONS = ("no_memory", "oracle", "memory")


def records_for(condition: str, task: Task, decisions_root: Path, records_mode: str = "current") -> list[Record]:
    all_records = load_records(decisions_root)
    if condition == "no_memory":
        return []
    if condition == "oracle":
        return current(all_records) if records_mode == "current" else all_records
    if condition == "memory":
        selected = for_files(all_records, task.files)
        if records_mode == "all":
            ids = {r.id for r in selected}
            selected = selected + [r for r in all_records if not r.is_rule and r.matches(task.files) and r.id not in ids]
        return selected
    raise ValueError(condition)


def render_context(records: list[Record]) -> str:
    if not records:
        return ""
    head = (
        "## Decision records\n\n"
        "Business decisions made outside the code. Only records with status\n"
        "`accepted` are rules. Where the code or docs disagree with an accepted\n"
        "record, the record is current. Follow the rules and the constraints;\n"
        "the rationale is there to help you interpret cases the rule does not\n"
        "spell out. A record grants no permission by itself.\n\n"
    )
    return head + "\n---\n\n".join(r.render() for r in records)
