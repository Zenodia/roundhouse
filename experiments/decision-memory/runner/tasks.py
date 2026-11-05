# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task specs: one folder per task with `task.yaml` and a hidden
`test_task.py` the agent never sees."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
TASKS_DIR = EXPERIMENT / "tasks"
FIXTURE_REPO = EXPERIMENT / "ledgerline"
REFERENCE_DIR = EXPERIMENT / "reference" / "ledgerline"


@dataclass
class Task:
    id: str
    title: str
    ladder: str
    decisions: list[str]
    files: list[str]
    prompt: str
    forbidden_patterns: list[str] = field(default_factory=list)
    dir: Path = Path(".")

    @property
    def test_file(self) -> Path:
        return self.dir / "test_task.py"


def load_tasks(ids: list[str] | None = None, root: Path = TASKS_DIR) -> list[Task]:
    out = []
    for d in sorted(root.iterdir()):
        spec = d / "task.yaml"
        if not spec.exists():
            continue
        y = yaml.safe_load(spec.read_text(encoding="utf-8"))
        t = Task(
            id=y["id"],
            title=y["title"],
            ladder=y.get("ladder", ""),
            decisions=list(y.get("decisions") or []),
            files=list(y.get("files") or []),
            prompt=y["prompt"].strip(),
            forbidden_patterns=list(y.get("forbidden_patterns") or []),
            dir=d,
        )
        if ids is None or t.id in ids or any(t.id.startswith(i) for i in ids):
            out.append(t)
    return out
