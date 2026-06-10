from webtoon_translator.core.models import RegionClass
from webtoon_translator.pipeline.detector import (
    RawBox,
    compute_tiles,
    containment,
    inset_bbox,
    link_regions,
    merge_boxes,
    reading_order,
)


def test_short_image_single_tile():
    assert compute_tiles(800, 1000) == [(0, 1000)]


def test_tall_strip_tiles_cover_and_overlap():
    tiles = compute_tiles(800, 20000)
    assert tiles[0][0] == 0
    assert tiles[-1][1] == 20000
    # full coverage, with overlap between consecutive tiles
    for (a0, a1), (b0, b1) in zip(tiles, tiles[1:]):
        assert b0 < a1, "tiles must overlap"
    # every tile has the same nominal height except possibly clamped at end
    heights = {t[1] - t[0] for t in tiles}
    assert len(heights) == 1


def test_tiles_no_taller_than_image():
    tiles = compute_tiles(800, 900)
    assert tiles == [(0, 900)]


def test_merge_boxes_across_seam():
    # same bubble seen in two tiles, split at the seam
    a = RawBox(bbox=(100, 950, 300, 1100), cls=RegionClass.BUBBLE, score=0.9)
    b = RawBox(bbox=(100, 1050, 300, 1250), cls=RegionClass.BUBBLE, score=0.8)
    merged = merge_boxes([a, b])
    assert len(merged) == 1
    assert merged[0].bbox == (100, 950, 300, 1250)
    assert merged[0].score == 0.9


def test_merge_keeps_distinct_boxes():
    a = RawBox(bbox=(0, 0, 100, 100), cls=RegionClass.BUBBLE, score=0.9)
    b = RawBox(bbox=(500, 500, 600, 600), cls=RegionClass.BUBBLE, score=0.8)
    assert len(merge_boxes([a, b])) == 2


def test_merge_respects_class():
    a = RawBox(bbox=(0, 0, 100, 100), cls=RegionClass.BUBBLE, score=0.9)
    b = RawBox(bbox=(0, 0, 100, 100), cls=RegionClass.TEXT_BUBBLE, score=0.8)
    assert len(merge_boxes([a, b])) == 2


def test_containment():
    assert containment((10, 10, 20, 20), (0, 0, 100, 100)) == 1.0
    assert containment((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_link_text_bubble_to_bubble():
    bubble = RawBox(bbox=(100, 100, 400, 300), cls=RegionClass.BUBBLE, score=0.95)
    text = RawBox(bbox=(150, 150, 350, 250), cls=RegionClass.TEXT_BUBBLE, score=0.9)
    free = RawBox(bbox=(500, 500, 700, 560), cls=RegionClass.TEXT_FREE, score=0.85)
    regions = link_regions([bubble, text, free])
    by_cls = {r.cls: r for r in regions}
    bubble_region = by_cls[RegionClass.BUBBLE]
    text_region = by_cls[RegionClass.TEXT_BUBBLE]
    free_region = by_cls[RegionClass.TEXT_FREE]
    assert text_region.parent_bubble_id == bubble_region.id
    assert text_region.render_bbox == inset_bbox(bubble_region.bbox, 0.10)
    assert free_region.parent_bubble_id is None
    assert free_region.render_bbox == free_region.bbox


def test_reading_order_top_to_bottom():
    r1 = link_regions([RawBox(bbox=(0, 1000, 100, 1100), cls=RegionClass.TEXT_FREE, score=1)])[0]
    r2 = link_regions([RawBox(bbox=(0, 100, 100, 200), cls=RegionClass.TEXT_FREE, score=1)])[0]
    ordered = reading_order([r1, r2])
    assert ordered[0].bbox[1] == 100
