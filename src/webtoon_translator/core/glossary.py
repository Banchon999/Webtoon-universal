"""Glossary import/export (CSV and JSON)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import GlossaryEntry


def load_glossary(path: Path) -> list[GlossaryEntry]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [GlossaryEntry.from_dict(d) for d in data]
    entries: list[GlossaryEntry] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0].strip():
                continue
            # header row detection
            if row[0].strip().lower() in ("source", "src", "term"):
                continue
            entries.append(
                GlossaryEntry(
                    source=row[0].strip(),
                    target=row[1].strip() if len(row) > 1 else "",
                    note=row[2].strip() if len(row) > 2 else "",
                )
            )
    return entries


def save_glossary(path: Path, entries: list[GlossaryEntry]) -> None:
    if path.suffix.lower() == ".json":
        path.write_text(
            json.dumps([e.to_dict() for e in entries], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "note"])
        for e in entries:
            writer.writerow([e.source, e.target, e.note])


def relevant_entries(entries: list[GlossaryEntry], texts: list[str]) -> list[GlossaryEntry]:
    """Glossary entries whose source term appears in any of `texts`."""
    blob = "\n".join(texts)
    return [e for e in entries if e.enabled and e.source and e.source in blob]
