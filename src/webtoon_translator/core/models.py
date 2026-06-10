"""Core data model: Project / Page / TextRegion and friends.

These dataclasses are the single source of truth flowing through the
pipeline and the GUI. They are JSON-serializable via to_dict/from_dict.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

BBox = tuple[int, int, int, int]  # x1, y1, x2, y2 (full-page pixel coords)


class RegionClass(StrEnum):
    BUBBLE = "bubble"
    TEXT_BUBBLE = "text_bubble"
    TEXT_FREE = "text_free"


class Stage(StrEnum):
    DETECT = "detect"
    OCR = "ocr"
    TRANSLATE = "translate"
    INPAINT = "inpaint"
    TYPESET = "typeset"


ALL_STAGES: tuple[Stage, ...] = (
    Stage.DETECT,
    Stage.OCR,
    Stage.TRANSLATE,
    Stage.INPAINT,
    Stage.TYPESET,
)


@dataclass
class FontSettings:
    family: str = "auto"  # "auto" -> pick by target-language script
    size: int = 0  # 0 = auto-fit
    color: str = "#000000"
    stroke_color: str = "#FFFFFF"
    stroke_width: int = 0
    align: str = "center"
    line_spacing: float = 1.15

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FontSettings:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TextRegion:
    bbox: BBox
    cls: RegionClass
    score: float = 1.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_bubble_id: str | None = None
    ocr_text: str = ""
    translation: str = ""
    ocr_edited: bool = False
    translation_edited: bool = False
    enabled: bool = True
    font: FontSettings = field(default_factory=FontSettings)
    render_bbox: BBox | None = None  # area used for typesetting (bubble interior)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cls"] = self.cls.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TextRegion:
        d = dict(d)
        d["bbox"] = tuple(d["bbox"])
        d["cls"] = RegionClass(d["cls"])
        if d.get("render_bbox"):
            d["render_bbox"] = tuple(d["render_bbox"])
        d["font"] = FontSettings.from_dict(d.get("font") or {})
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Page:
    source_path: Path
    size: tuple[int, int] = (0, 0)  # (width, height)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    regions: list[TextRegion] = field(default_factory=list)
    stages_done: list[Stage] = field(default_factory=list)
    cleaned_image_path: Path | None = None
    rendered_image_path: Path | None = None

    def mark_done(self, stage: Stage) -> None:
        if stage not in self.stages_done:
            self.stages_done.append(stage)

    def invalidate(self, stage: Stage) -> None:
        """Drop `stage` and everything after it from the done-list."""
        order = list(ALL_STAGES)
        idx = order.index(stage)
        invalid = set(order[idx:])
        self.stages_done = [s for s in self.stages_done if s not in invalid]

    def text_regions(self) -> list[TextRegion]:
        """Regions that carry text (OCR/translate/typeset targets)."""
        return [
            r
            for r in self.regions
            if r.enabled and r.cls in (RegionClass.TEXT_BUBBLE, RegionClass.TEXT_FREE)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_path": str(self.source_path),
            "size": list(self.size),
            "regions": [r.to_dict() for r in self.regions],
            "stages_done": [s.value for s in self.stages_done],
            "cleaned_image_path": str(self.cleaned_image_path) if self.cleaned_image_path else None,
            "rendered_image_path": str(self.rendered_image_path) if self.rendered_image_path else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Page:
        return cls(
            id=d["id"],
            source_path=Path(d["source_path"]),
            size=tuple(d.get("size") or (0, 0)),
            regions=[TextRegion.from_dict(r) for r in d.get("regions", [])],
            stages_done=[Stage(s) for s in d.get("stages_done", [])],
            cleaned_image_path=Path(d["cleaned_image_path"]) if d.get("cleaned_image_path") else None,
            rendered_image_path=Path(d["rendered_image_path"]) if d.get("rendered_image_path") else None,
        )


@dataclass
class GlossaryEntry:
    source: str
    target: str
    note: str = ""
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GlossaryEntry:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Project:
    pages: list[Page] = field(default_factory=list)
    source_lang: str = "auto"
    target_lang: str = "th"
    glossary: list[GlossaryEntry] = field(default_factory=list)
    openrouter_model: str = "google/gemini-2.5-flash"
    version: int = 1

    def page_by_id(self, page_id: str) -> Page | None:
        return next((p for p in self.pages if p.id == page_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "openrouter_model": self.openrouter_model,
            "glossary": [g.to_dict() for g in self.glossary],
            "pages": [p.to_dict() for p in self.pages],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Project:
        return cls(
            version=d.get("version", 1),
            source_lang=d.get("source_lang", "auto"),
            target_lang=d.get("target_lang", "th"),
            openrouter_model=d.get("openrouter_model", "google/gemini-2.5-flash"),
            glossary=[GlossaryEntry.from_dict(g) for g in d.get("glossary", [])],
            pages=[Page.from_dict(p) for p in d.get("pages", [])],
        )
