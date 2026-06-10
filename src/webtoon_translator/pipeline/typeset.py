"""Typesetting: fit translated text into bubbles and render with Pillow.

Thai has no inter-word spaces, so line-break opportunities come from
pythainlp's newmm tokenizer; CJK falls back to per-character breaks; other
languages break on spaces. Font size is found by binary search so the wrapped
block fits the render box.
"""

from __future__ import annotations

import functools
import json
import logging
import re

from PIL import Image, ImageDraw, ImageFont

from ..core.models import FontSettings, RegionClass, TextRegion
from ..paths import fonts_dir

log = logging.getLogger(__name__)

MAX_FONT_SIZE = 72
MIN_FONT_SIZE = 11

CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
THAI_RE = re.compile(r"[฀-๿]")


@functools.lru_cache(maxsize=1)
def font_registry() -> dict:
    """fonts.json: {"fonts": [{"name", "file", "bold_file", "scripts": ["thai", ...]}], "defaults": {...}}"""
    path = fonts_dir() / "fonts.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"fonts": [], "defaults": {}}


def pick_font_file(family: str, target_lang: str) -> str:
    """Resolve a font family name (or "auto") to a TTF path."""
    reg = font_registry()
    fonts = {f["name"]: f for f in reg.get("fonts", [])}
    if family != "auto" and family in fonts:
        return str(fonts_dir() / fonts[family]["file"])
    default_name = reg.get("defaults", {}).get(target_lang) or reg.get("defaults", {}).get("*")
    if default_name and default_name in fonts:
        return str(fonts_dir() / fonts[default_name]["file"])
    if fonts:
        return str(fonts_dir() / next(iter(fonts.values()))["file"])
    raise FileNotFoundError("no bundled fonts found (assets/fonts/fonts.json)")


def break_units(text: str, lang: str) -> list[str]:
    """Split text into the smallest units between which a line break is legal."""
    if THAI_RE.search(text):
        try:
            from pythainlp.tokenize import word_tokenize

            units: list[str] = []
            for chunk in text.split(" "):
                if not chunk:
                    continue
                if THAI_RE.search(chunk):
                    units.extend(t for t in word_tokenize(chunk, engine="newmm") if t.strip())
                else:
                    units.append(chunk)
            return units
        except Exception as e:  # pythainlp missing dict data etc.
            log.warning("thai tokenization failed (%s); falling back to chars", e)
            return list(text)
    if CJK_RE.search(text) and " " not in text.strip():
        return [c for c in text if not c.isspace()]
    return text.split()


def _joiner(units: list[str]) -> str:
    """Whether units should be joined with spaces when measuring/rendering."""
    sample = "".join(units)
    if THAI_RE.search(sample) or (CJK_RE.search(sample) and len(units) > 1 and len(units[0]) <= 2):
        return ""
    return " "


def wrap_units(units: list[str], font: ImageFont.FreeTypeFont, max_width: int, joiner: str) -> list[str]:
    lines: list[str] = []
    current = ""
    for unit in units:
        candidate = unit if not current else current + joiner + unit
        if font.getlength(candidate) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = unit
    if current:
        lines.append(current)
    return lines


def fit_text(
    text: str,
    box_w: int,
    box_h: int,
    font_path: str,
    lang: str,
    line_spacing: float = 1.15,
    max_size: int = MAX_FONT_SIZE,
    min_size: int = MIN_FONT_SIZE,
) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """Binary-search the largest font size whose wrapped block fits box_w x box_h."""
    units = break_units(text, lang)
    joiner = _joiner(units)

    def layout(size: int) -> tuple[ImageFont.FreeTypeFont, list[str], bool]:
        font = ImageFont.truetype(font_path, size)
        lines = wrap_units(units, font, box_w, joiner)
        ascent, descent = font.getmetrics()
        line_h = (ascent + descent) * line_spacing
        height = line_h * len(lines)
        width_ok = all(font.getlength(line) <= box_w for line in lines)
        return font, lines, width_ok and height <= box_h

    lo, hi = min_size, max_size
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        font, lines, fits = layout(mid)
        if fits:
            best = (font, lines)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        font, lines, _ = layout(min_size)
        best = (font, lines)
    return best


def render_region(canvas: Image.Image, region: TextRegion, target_lang: str) -> None:
    text = region.translation.strip()
    if not text or not region.enabled:
        return
    box = region.render_bbox or region.bbox
    x1, y1, x2, y2 = box
    box_w, box_h = max(8, x2 - x1), max(8, y2 - y1)
    fs: FontSettings = region.font
    font_path = pick_font_file(fs.family, target_lang)

    if fs.size > 0:
        font = ImageFont.truetype(font_path, fs.size)
        units = break_units(text, target_lang)
        lines = wrap_units(units, font, box_w, _joiner(units))
    else:
        font, lines = fit_text(text, box_w, box_h, font_path, target_lang, fs.line_spacing)

    stroke_width = fs.stroke_width
    if stroke_width == 0 and region.cls == RegionClass.TEXT_FREE:
        stroke_width = max(2, font.size // 10)  # free text needs an outline over art

    draw = ImageDraw.Draw(canvas)
    ascent, descent = font.getmetrics()
    line_h = (ascent + descent) * fs.line_spacing
    total_h = line_h * len(lines)
    y = y1 + (box_h - total_h) / 2
    cx = x1 + box_w / 2
    for line in lines:
        lw = draw.textlength(line, font=font)
        if fs.align == "left":
            lx = x1
        elif fs.align == "right":
            lx = x2 - lw
        else:
            lx = cx - lw / 2
        draw.text(
            (lx, y),
            line,
            font=font,
            fill=fs.color,
            stroke_width=stroke_width,
            stroke_fill=fs.stroke_color if stroke_width else None,
        )
        y += line_h


def render_page(cleaned: Image.Image, regions: list[TextRegion], target_lang: str) -> Image.Image:
    canvas = cleaned.convert("RGB").copy()
    for region in regions:
        if region.cls == RegionClass.BUBBLE:
            continue
        render_region(canvas, region, target_lang)
    return canvas
