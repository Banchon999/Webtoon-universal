#!/usr/bin/env python3
"""Generate tests/data/sample_strip.png — a synthetic webtoon strip with
speech bubbles and text, used by the e2e pipeline test."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tests" / "data" / "sample_strip.png"

W, H = 800, 3000


def bubble(draw: ImageDraw.ImageDraw, cx: int, cy: int, rx: int, ry: int, text: str, font) -> None:
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill="white", outline="black", width=4)
    lines = text.split("\n")
    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center")
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.multiline_text((cx - tw / 2, cy - th / 2), text, font=font, fill="black", align="center")
    _ = lines


def main() -> int:
    img = Image.new("RGB", (W, H), (90, 120, 200))
    draw = ImageDraw.Draw(img)
    # background "art": gradient blocks and shapes
    for i in range(0, H, 250):
        color = (60 + (i * 7) % 120, 90 + (i * 13) % 100, 150 + (i * 5) % 90)
        draw.rectangle((0, i, W, i + 250), fill=color)
    for i in range(40):
        x, y = (i * 173) % W, (i * 631) % H
        draw.ellipse((x, y, x + 60, y + 60), fill=(255, 220, 120))

    font_path = REPO / "assets" / "fonts" / "NotoSans-Regular.ttf"
    font = ImageFont.truetype(str(font_path), 28)

    bubble(draw, 250, 400, 200, 110, "HELLO THERE!\nNICE WEATHER", font)
    bubble(draw, 550, 1300, 190, 100, "WHERE ARE\nYOU GOING?", font)
    bubble(draw, 300, 2300, 210, 120, "I MUST FIND\nTHE HERO", font)
    # free text (no bubble)
    draw.text((250, 2800), "MEANWHILE...", font=ImageFont.truetype(str(font_path), 40), fill="white",
              stroke_width=3, stroke_fill="black")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
