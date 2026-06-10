import numpy as np

from webtoon_translator.core.models import RegionClass, TextRegion
from webtoon_translator.pipeline.inpaint import (
    LAMA_SIZE,
    build_mask,
    is_uniform_region,
    mask_windows,
)


def region(bbox, cls=RegionClass.TEXT_BUBBLE):
    return TextRegion(bbox=bbox, cls=cls)


def test_build_mask_covers_dilated_boxes():
    mask = build_mask((500, 500), [region((100, 100, 200, 150))])
    assert mask[120, 150] == 255
    assert mask[95, 150] == 255  # dilation reaches slightly outside
    assert mask[400, 400] == 0


def test_build_mask_skips_bubble_class_and_disabled():
    r1 = region((10, 10, 50, 50), cls=RegionClass.BUBBLE)
    r2 = region((100, 100, 150, 150))
    r2.enabled = False
    mask = build_mask((200, 200), [r1, r2])
    assert mask.max() == 0


def test_mask_windows_within_bounds():
    mask = build_mask((4000, 800), [region((100, 1000, 300, 1100)), region((400, 3500, 700, 3600))])
    wins = mask_windows(mask)
    assert wins
    for x, y, w, h in wins:
        assert 0 <= x and x + w <= 800
        assert 0 <= y and y + h <= 4000
        assert w <= LAMA_SIZE and h <= LAMA_SIZE


def test_mask_windows_cover_all_mask_pixels():
    mask = build_mask((4000, 800), [region((100, 1000, 300, 1100)), region((400, 3500, 700, 3600))])
    covered = np.zeros_like(mask)
    for x, y, w, h in mask_windows(mask):
        covered[y : y + h, x : x + w] = 255
    assert np.all(covered[mask > 0] == 255)


def test_mask_windows_small_image():
    mask = build_mask((300, 300), [region((50, 50, 250, 250))])
    wins = mask_windows(mask)
    for x, y, w, h in wins:
        assert w <= 300 and h <= 300


def test_is_uniform_region_on_white_bubble():
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    r = region((100, 100, 300, 200))
    uniform, color = is_uniform_region(img, r)
    assert uniform
    assert np.allclose(color, 255)


def test_is_uniform_region_on_noisy_art():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (400, 400, 3), dtype=np.uint8).astype(np.uint8)
    r = region((100, 100, 300, 200))
    uniform, _ = is_uniform_region(img, r)
    assert not uniform
