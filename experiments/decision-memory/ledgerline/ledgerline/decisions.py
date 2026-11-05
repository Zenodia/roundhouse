"""Decision records: the digital copy of decisions made outside the code.

A record is a Markdown file under `decisions/` with a YAML front matter.
This module reads them, answers "which rules are current", selects the
records relevant to a set of files, and validates the folder's consistency.
Policy tests read their expected values from here, so when a decision
changes, the tests change with it and nothing else has to.

Only `status: accepted` records are rules. `proposed` is not yet a decision,
`rejected` never was one, `superseded` was one and is kept for history.
A record informs; it grants no permission by itself.

Usage:
    python -m ledgerline.decisions validate
    python -m ledgerline.decisions current
    python -m ledgerline.decisions for ledgerline/reports.py
"""

from __future__ import annotations

import fnmatch
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

STATUSES = ("proposed", "accepted", "superseded", "rejected")
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "decisions"


@dataclass
class Record:
    id: str
    title: str
    status: str
    owner: str
    decided_on: str
    effective_from: str | None
    supersedes: str | None
    superseded_by: str | None
    scope: list[str]
    tags: list[str]
    parameters: dict[str, Any]
    source: str | None
    body: str
    path: Path
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_rule(self) -> bool:
        return self.status == "accepted"

    def matches(self, files: list[str]) -> bool:
        return any(fnmatch.fnmatch(f, pattern) for f in files for pattern in self.scope)

    def render(self) -> str:
        """The record as the agent sees it: front matter facts, then body."""
        head = [
            f"# {self.id}: {self.title}",
            f"status: {self.status} | owner: {self.owner} | decided: {self.decided_on}"
            + (f" | effective: {self.effective_from}" if self.effective_from else ""),
        ]
        if self.supersedes:
            head.append(f"supersedes: {self.supersedes}")
        if self.superseded_by:
            head.append(f"superseded by: {self.superseded_by}")
        head.append("scope: " + ", ".join(self.scope))
        if self.parameters:
            head.append("parameters: " + yaml.safe_dump(self.parameters, default_flow_style=True).strip())
        return "\n".join(head) + "\n\n" + self.body.strip() + "\n"


def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        raise ValueError("record has no front matter")
    _, fm, body = text.split("---", 2)
    return yaml.safe_load(fm) or {}, body


def load_record(path: Path) -> Record:
    fm, body = _split_front_matter(path.read_text(encoding="utf-8"))
    known = {"id", "title", "status", "owner", "decided_on", "effective_from", "supersedes", "superseded_by", "scope", "tags", "parameters", "source"}
    return Record(
        id=str(fm["id"]),
        title=str(fm["title"]),
        status=str(fm["status"]),
        owner=str(fm.get("owner", "")),
        decided_on=str(fm.get("decided_on", "")),
        effective_from=None if fm.get("effective_from") is None else str(fm["effective_from"]),
        supersedes=fm.get("supersedes"),
        superseded_by=fm.get("superseded_by"),
        scope=list(fm.get("scope") or []),
        tags=list(fm.get("tags") or []),
        parameters=dict(fm.get("parameters") or {}),
        source=fm.get("source"),
        body=body,
        path=path,
        extra={k: v for k, v in fm.items() if k not in known},
    )


def load_records(root: Path | str = DEFAULT_ROOT) -> list[Record]:
    root = Path(root)
    return sorted((load_record(p) for p in root.glob("DR-*.md")), key=lambda r: r.id)


def current(records: list[Record]) -> list[Record]:
    return [r for r in records if r.is_rule]


def for_files(records: list[Record], files: list[str]) -> list[Record]:
    """Current records whose scope covers any of `files`."""
    return [r for r in current(records) if r.matches(files)]


def param(key: str, default: Any = None, root: Path | str = DEFAULT_ROOT) -> Any:
    """The value of a parameter from the one accepted record that defines it."""
    owners = [r for r in current(load_records(root)) if key in r.parameters]
    if not owners:
        return default
    if len(owners) > 1:
        raise ValueError(f"parameter {key!r} is defined by several accepted records: {[r.id for r in owners]}")
    return owners[0].parameters[key]


def validate(records: list[Record]) -> list[str]:
    """Problems in the folder, as sentences. Empty means consistent."""
    problems: list[str] = []
    by_id = {r.id: r for r in records}
    if len(by_id) != len(records):
        problems.append("duplicate record ids")
    for r in records:
        if r.status not in STATUSES:
            problems.append(f"{r.id}: unknown status {r.status!r}")
        if not r.owner:
            problems.append(f"{r.id}: no owner")
        if r.is_rule and not r.scope:
            problems.append(f"{r.id}: accepted record with empty scope")
        if r.supersedes:
            old = by_id.get(r.supersedes)
            if old is None:
                problems.append(f"{r.id}: supersedes unknown {r.supersedes}")
            elif r.is_rule and old.status != "superseded":
                problems.append(f"{r.id} is accepted but {old.id} is still {old.status}")
            elif old.superseded_by != r.id and r.is_rule:
                problems.append(f"{old.id} should name superseded_by: {r.id}")
        if r.status == "superseded" and not r.superseded_by:
            problems.append(f"{r.id}: superseded by nothing")
    seen: dict[str, str] = {}
    for r in current(records):
        for key in r.parameters:
            if key in seen:
                problems.append(f"parameter {key!r} set by both {seen[key]} and {r.id}")
            seen[key] = r.id
    return problems


def _set_front_matter(path: Path, **changes: Any) -> None:
    text = path.read_text(encoding="utf-8")
    _, fm, body = text.split("---", 2)
    data = yaml.safe_load(fm) or {}
    data.update(changes)
    path.write_text("---\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True) + "---" + body, encoding="utf-8")


def accept(record_id: str, root: Path | str = DEFAULT_ROOT) -> list[str]:
    """The owner's approval, applied consistently: the record becomes
    `accepted`, and the record it supersedes becomes `superseded` with the
    back-link. Returns the paths changed. Refuses if the folder would be
    inconsistent afterwards (two accepted records setting one parameter)."""
    records = load_records(root)
    by_id = {r.id: r for r in records}
    rec = by_id[record_id]
    if rec.status != "proposed":
        raise ValueError(f"{record_id} is {rec.status}, not proposed")
    changed = []
    _set_front_matter(rec.path, status="accepted")
    changed.append(str(rec.path))
    if rec.supersedes:
        old = by_id[rec.supersedes]
        _set_front_matter(old.path, status="superseded", superseded_by=record_id)
        changed.append(str(old.path))
    problems = validate(load_records(root))
    if problems:
        # Roll back rather than leave the folder in a state `param()` refuses.
        _set_front_matter(rec.path, status="proposed")
        if rec.supersedes:
            _set_front_matter(by_id[rec.supersedes].path, status=by_id[rec.supersedes].status, superseded_by=by_id[rec.supersedes].superseded_by)
        raise ValueError("accepting would leave the folder inconsistent: " + "; ".join(problems))
    return changed


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "validate"
    root = DEFAULT_ROOT
    if cmd == "accept":
        for path in accept(argv[1], root):
            print("updated", path)
        return 0
    records = load_records(root)
    if cmd == "validate":
        problems = validate(records)
        for p in problems:
            print("PROBLEM:", p)
        print(f"{len(records)} records, {len(current(records))} accepted, {len(problems)} problems")
        return 1 if problems else 0
    if cmd == "current":
        for r in current(records):
            print(f"{r.id}  {r.title}  [{r.owner}]  params={r.parameters}")
        return 0
    if cmd == "for":
        for r in for_files(records, argv[1:]):
            print(r.render())
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
