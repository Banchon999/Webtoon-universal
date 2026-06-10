"""Project save/load.

A .wtproj file is a zip archive:
    project.json   - serialized Project
    cleaned/       - cached inpainted page PNGs
    rendered/      - typeset page PNGs

Cached images referenced by Page.cleaned_image_path / rendered_image_path are
re-pointed into a per-project work directory on load.
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path

from .models import Page, Project

PROJECT_JSON = "project.json"


def project_work_dir(project_path: Path) -> Path:
    """Directory next to the project file holding extracted cached images."""
    d = project_path.with_suffix(".wtproj.work")
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_project(project: Project, path: Path) -> None:
    """Atomic save: write to temp file then rename over target."""
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(suffix=".wtproj.tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for page in project.pages:
                for attr, folder in (("cleaned_image_path", "cleaned"), ("rendered_image_path", "rendered")):
                    p: Path | None = getattr(page, attr)
                    if p and Path(p).exists():
                        zf.write(p, f"{folder}/{page.id}.png")
            zf.writestr(PROJECT_JSON, json.dumps(project.to_dict(), ensure_ascii=False, indent=1))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_project(path: Path) -> Project:
    path = Path(path)
    work = project_work_dir(path)
    with zipfile.ZipFile(path) as zf:
        project = Project.from_dict(json.loads(zf.read(PROJECT_JSON)))
        names = set(zf.namelist())
        for page in project.pages:
            page.cleaned_image_path = _extract_cached(zf, names, "cleaned", page, work)
            page.rendered_image_path = _extract_cached(zf, names, "rendered", page, work)
    return project


def _extract_cached(
    zf: zipfile.ZipFile, names: set[str], folder: str, page: Page, work: Path
) -> Path | None:
    member = f"{folder}/{page.id}.png"
    if member not in names:
        return None
    out = work / folder / f"{page.id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(zf.read(member))
    return out
