"""Export rendered pages to image files."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from ..core.models import Page, Project

log = logging.getLogger(__name__)

FORMATS = {
    "png": {"ext": ".png", "kwargs": {"optimize": True}},
    "jpg": {"ext": ".jpg", "kwargs": {"quality": 92}},
    "webp": {"ext": ".webp", "kwargs": {"quality": 92}},
}


def export_page(page: Page, out_dir: Path, fmt: str = "png", quality: int | None = None) -> Path | None:
    src = page.rendered_image_path or page.cleaned_image_path
    if not src or not Path(src).exists():
        log.warning("page %s has no rendered output; skipping", page.source_path.name)
        return None
    spec = FORMATS[fmt]
    kwargs = dict(spec["kwargs"])
    if quality is not None and "quality" in kwargs:
        kwargs["quality"] = quality
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (Path(page.source_path).stem + "_translated" + spec["ext"])
    img = Image.open(src)
    if fmt == "jpg":
        img = img.convert("RGB")
    img.save(out, **kwargs)
    return out


def export_project(project: Project, out_dir: Path, fmt: str = "png", quality: int | None = None) -> list[Path]:
    results = []
    for page in project.pages:
        p = export_page(page, out_dir, fmt, quality)
        if p:
            results.append(p)
    return results
