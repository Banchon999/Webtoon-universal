from pathlib import Path

from PIL import Image

from webtoon_translator.core.glossary import load_glossary, save_glossary
from webtoon_translator.core.models import (
    FontSettings,
    GlossaryEntry,
    Page,
    Project,
    RegionClass,
    Stage,
    TextRegion,
)
from webtoon_translator.core.project_io import load_project, save_project


def make_project(tmp_path: Path) -> Project:
    img = tmp_path / "page1.png"
    Image.new("RGB", (100, 100), "white").save(img)
    rendered = tmp_path / "rendered.png"
    Image.new("RGB", (100, 100), "red").save(rendered)
    region = TextRegion(
        bbox=(10, 10, 90, 50),
        cls=RegionClass.TEXT_BUBBLE,
        ocr_text="안녕",
        translation="สวัสดี",
        font=FontSettings(size=20, color="#112233"),
    )
    page = Page(source_path=img, size=(100, 100), regions=[region])
    page.rendered_image_path = rendered
    page.mark_done(Stage.DETECT)
    page.mark_done(Stage.OCR)
    return Project(
        pages=[page],
        target_lang="th",
        glossary=[GlossaryEntry(source="안녕", target="สวัสดี", note="greeting")],
    )


def test_roundtrip(tmp_path):
    project = make_project(tmp_path)
    path = tmp_path / "test.wtproj"
    save_project(project, path)
    loaded = load_project(path)

    assert loaded.target_lang == "th"
    assert loaded.glossary[0].source == "안녕"
    page = loaded.pages[0]
    assert page.size == (100, 100)
    assert page.stages_done == [Stage.DETECT, Stage.OCR]
    region = page.regions[0]
    assert region.cls == RegionClass.TEXT_BUBBLE
    assert region.bbox == (10, 10, 90, 50)
    assert region.translation == "สวัสดี"
    assert region.font.color == "#112233"
    # cached rendered image extracted
    assert page.rendered_image_path and page.rendered_image_path.exists()


def test_stage_invalidation():
    page = Page(source_path=Path("x.png"))
    for s in (Stage.DETECT, Stage.OCR, Stage.TRANSLATE, Stage.TYPESET):
        page.mark_done(s)
    page.invalidate(Stage.OCR)
    assert page.stages_done == [Stage.DETECT]


def test_glossary_csv_roundtrip(tmp_path):
    entries = [GlossaryEntry(source="용사", target="ผู้กล้า", note="hero")]
    path = tmp_path / "g.csv"
    save_glossary(path, entries)
    loaded = load_glossary(path)
    assert loaded[0].source == "용사"
    assert loaded[0].target == "ผู้กล้า"
    assert loaded[0].note == "hero"


def test_glossary_json_roundtrip(tmp_path):
    entries = [GlossaryEntry(source="마왕", target="จอมมาร", enabled=False)]
    path = tmp_path / "g.json"
    save_glossary(path, entries)
    loaded = load_glossary(path)
    assert loaded[0].enabled is False
