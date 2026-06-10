from PIL import Image

from webtoon_translator.core.models import FontSettings, RegionClass, TextRegion
from webtoon_translator.pipeline.typeset import (
    break_units,
    fit_text,
    pick_font_file,
    render_page,
    wrap_units,
)


def test_break_units_english():
    assert break_units("HELLO BIG WORLD", "en") == ["HELLO", "BIG", "WORLD"]


def test_break_units_thai():
    units = break_units("สวัสดีครับผม", "th")
    assert len(units) > 1
    assert "".join(units) == "สวัสดีครับผม"


def test_break_units_cjk():
    units = break_units("こんにちは世界", "ja")
    assert len(units) == 7


def test_pick_font_thai_vs_default():
    thai = pick_font_file("auto", "th")
    latin = pick_font_file("auto", "en")
    assert "Sarabun" in thai
    assert "NotoSans" in latin


def test_fit_text_fits_box():
    font_path = pick_font_file("auto", "en")
    font, lines = fit_text("HELLO WORLD THIS IS A SPEECH BUBBLE", 200, 120, font_path, "en")
    assert lines
    assert all(font.getlength(line) <= 200 for line in lines)


def test_fit_text_thai():
    font_path = pick_font_file("auto", "th")
    font, lines = fit_text("ผู้กล้าออกเดินทางไปยังหมู่บ้านแห่งหนึ่งที่อยู่ไกลมาก", 180, 100, font_path, "th")
    assert lines
    assert all(font.getlength(line) <= 180 for line in lines)


def test_wrap_units_never_empty_lines():
    font_path = pick_font_file("auto", "en")
    from PIL import ImageFont

    font = ImageFont.truetype(font_path, 20)
    lines = wrap_units(["supercalifragilistic"], font, 30, " ")
    assert lines == ["supercalifragilistic"]  # single unit wider than box still emitted


def test_render_page_draws_translation():
    img = Image.new("RGB", (400, 300), "white")
    region = TextRegion(
        bbox=(50, 50, 350, 250),
        cls=RegionClass.TEXT_BUBBLE,
        translation="สวัสดีชาวโลก",
        font=FontSettings(),
    )
    region.render_bbox = region.bbox
    out = render_page(img, [region], "th")
    # some pixels must change (text drawn)
    assert out.tobytes() != img.tobytes()


def test_render_page_skips_disabled():
    img = Image.new("RGB", (400, 300), "white")
    region = TextRegion(
        bbox=(50, 50, 350, 250), cls=RegionClass.TEXT_BUBBLE, translation="HELLO", enabled=False
    )
    out = render_page(img, [region], "en")
    assert out.tobytes() == img.tobytes()
