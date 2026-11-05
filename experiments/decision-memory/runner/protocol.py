# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The patch protocol: what the model is shown and how its answer is read.

Two parts, kept apart on purpose so Roundhouse can cache the first:

- `instructions` (the system prompt): the stable repo context -- the
  service reference, the core modules, and the decision records for this
  condition. Identical across tasks within a condition, so it is one
  shared prefix.
- `input` (the user turn): the task ticket and the current contents of the
  files it touches.

The model answers with whole files:

    <<<FILE ledgerline/reports.py>>>
    ...complete new content...
    <<<END FILE>>>
"""

from __future__ import annotations

import re
from pathlib import Path

from .tasks import Task

CORE_FILES = ("ledgerline/models.py", "ledgerline/storage.py", "ledgerline/payments.py")

SYSTEM_PREAMBLE = """You are a senior engineer working in the Ledgerline payments service, a small Python repository.
You will receive a ticket. Implement it by returning complete files.

Rules:
- Return every file you create or change as a block:
  <<<FILE path/from/repo/root.py>>>
  <complete file content>
  <<<END FILE>>>
- Return the full content of each file, not a diff. Files you do not return are left unchanged.
- Do not modify tests/, decisions/ or docs/. Do not add dependencies.
- Keep existing public functions working unless the ticket says otherwise.
- Say nothing outside the file blocks except, optionally, one short paragraph of notes before them.
"""

FILE_RE = re.compile(r"<<<FILE\s+(?P<path>[^>\n]+?)\s*>>>\n(?P<body>.*?)\n?<<<END FILE>>>", re.S)


def _read(repo: Path, rel: str) -> str | None:
    p = repo / rel
    return p.read_text(encoding="utf-8") if p.exists() else None


def _fence(rel: str, content: str | None) -> str:
    if content is None:
        return f"### {rel}\n(does not exist yet)\n"
    return f"### {rel}\n```python\n{content.rstrip()}\n```\n"


def build_instructions(repo: Path, decision_context: str) -> str:
    parts = [SYSTEM_PREAMBLE, "## Service reference (docs/SERVICE.md)\n\n" + (_read(repo, "docs/SERVICE.md") or "")]
    parts.append("## Core modules\n\n" + "\n".join(_fence(f, _read(repo, f)) for f in CORE_FILES))
    if decision_context:
        parts.append(decision_context)
    return "\n\n".join(parts)


def build_input(repo: Path, task: Task) -> str:
    files = [f for f in task.files if f not in CORE_FILES]
    shown = "\n".join(_fence(f, _read(repo, f)) for f in files) if files else "(the files you need are in the core modules above)\n"
    return f"## Ticket {task.id}: {task.title}\n\n{task.prompt}\n\n## Files this ticket touches\n\n{shown}\nReturn the complete files."


def parse_files(answer: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in FILE_RE.finditer(answer):
        path = m.group("path").strip().strip("`")
        body = m.group("body")
        # Tolerate a model that wraps the body in a code fence.
        fenced = re.match(r"^```[a-zA-Z0-9_-]*\n(.*)\n```\s*$", body, re.S)
        if fenced:
            body = fenced.group(1)
        out[path] = body.rstrip() + "\n"
    return out


def safe_to_write(path: str) -> bool:
    p = Path(path)
    if p.is_absolute() or ".." in p.parts:
        return False
    top = p.parts[0] if p.parts else ""
    return top == "ledgerline" and p.suffix == ".py"
