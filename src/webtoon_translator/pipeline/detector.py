"""Text & bubble detection with ogkalu/comic-text-and-bubble-detector (RT-DETR-v2).

Webtoon strips are extremely tall, while the model was trained on ~square
640px images (tall sources were split vertically during training). We
therefore run detection over overlapping horizontal tiles and merge boxes
across tile seams.

Tile math and box merging are pure functions, unit-testable without models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..core.models import BBox, RegionClass, TextRegion

log = logging.getLogger(__name__)

# Aspect ratio above which we switch to tiled detection
TILE_ASPECT_THRESHOLD = 1.5
# Tile height as multiple of width (roughly square-ish after 640 resize)
TILE_HEIGHT_FACTOR = 1.4
MAX_TILE_H = 1792
OVERLAP_FRACTION = 0.25


@dataclass(frozen=True)
class RawBox:
    bbox: BBox
    cls: RegionClass
    score: float


def compute_tiles(width: int, height: int) -> list[tuple[int, int]]:
    """Return (y0, y1) tile spans covering the image with overlap."""
    if height <= int(width * TILE_ASPECT_THRESHOLD):
        return [(0, height)]
    tile_h = min(int(width * TILE_HEIGHT_FACTOR), MAX_TILE_H)
    tile_h = min(tile_h, height)
    step = max(1, int(tile_h * (1 - OVERLAP_FRACTION)))
    tiles: list[tuple[int, int]] = []
    y = 0
    while True:
        y1 = min(y + tile_h, height)
        tiles.append((max(0, y1 - tile_h), y1))
        if y1 >= height:
            break
        y += step
    return tiles


def _iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _intersection_over_min(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    min_area = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1, min_area)


def _union(a: BBox, b: BBox) -> BBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _seam_split(a: BBox, b: BBox) -> bool:
    """Same bubble cut at a tile seam: near-identical x-span, partial y-overlap."""
    x_overlap = min(a[2], b[2]) - max(a[0], b[0])
    if x_overlap <= 0:
        return False
    min_w = min(a[2] - a[0], b[2] - b[0])
    if x_overlap / max(1, min_w) < 0.85:
        return False
    y_overlap = min(a[3], b[3]) - max(a[1], b[1])
    min_h = min(a[3] - a[1], b[3] - b[1])
    return y_overlap / max(1, min_h) > 0.15


def merge_boxes(boxes: list[RawBox], iou_thr: float = 0.5, iom_thr: float = 0.8) -> list[RawBox]:
    """Greedy same-class merge of boxes split across tile seams, then NMS-like dedup.

    Two same-class boxes are merged (union, max score) when IoU > iou_thr,
    intersection-over-min-area > iom_thr, or they look like one box cut at a
    tile seam (matching x-span with partial vertical overlap).
    """
    boxes = sorted(boxes, key=lambda b: -b.score)
    merged: list[RawBox] = []
    for box in boxes:
        absorbed = False
        for i, kept in enumerate(merged):
            if kept.cls != box.cls:
                continue
            if (
                _iou(kept.bbox, box.bbox) > iou_thr
                or _intersection_over_min(kept.bbox, box.bbox) > iom_thr
                or _seam_split(kept.bbox, box.bbox)
            ):
                merged[i] = RawBox(
                    bbox=_union(kept.bbox, box.bbox),
                    cls=kept.cls,
                    score=max(kept.score, box.score),
                )
                absorbed = True
                break
        if not absorbed:
            merged.append(box)
    return merged


def containment(inner: BBox, outer: BBox) -> float:
    """Fraction of `inner` area inside `outer`."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def link_regions(raw: list[RawBox]) -> list[TextRegion]:
    """Convert raw boxes to TextRegions and link text_bubble boxes to their bubbles.

    A linked text region gets the (inset) bubble interior as its render_bbox.
    """
    bubbles = [b for b in raw if b.cls == RegionClass.BUBBLE]
    regions: list[TextRegion] = []
    bubble_regions = [TextRegion(bbox=b.bbox, cls=b.cls, score=b.score) for b in bubbles]
    regions.extend(bubble_regions)
    for b in raw:
        if b.cls == RegionClass.BUBBLE:
            continue
        region = TextRegion(bbox=b.bbox, cls=b.cls, score=b.score)
        if b.cls == RegionClass.TEXT_BUBBLE:
            best, best_cov = None, 0.7
            for bub in bubble_regions:
                cov = containment(b.bbox, bub.bbox)
                if cov >= best_cov:
                    best, best_cov = bub, cov
            if best is not None:
                region.parent_bubble_id = best.id
                region.render_bbox = inset_bbox(best.bbox, 0.10)
        if region.render_bbox is None:
            region.render_bbox = region.bbox
        regions.append(region)
    return regions


def inset_bbox(bbox: BBox, fraction: float) -> BBox:
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    dx, dy = int(w * fraction), int(h * fraction)
    return (bbox[0] + dx, bbox[1] + dy, bbox[2] - dx, bbox[3] - dy)


def reading_order(regions: list[TextRegion], rtl: bool = False) -> list[TextRegion]:
    """Sort text regions top-to-bottom; ties (same band) left-to-right or right-to-left."""

    def key(r: TextRegion):
        cy = (r.bbox[1] + r.bbox[3]) / 2
        cx = (r.bbox[0] + r.bbox[2]) / 2
        band = cy // 200  # coarse vertical banding
        return (band, -cx if rtl else cx, cy)

    return sorted(regions, key=key)


class OnnxBubbleDetector:
    """Default detector: ogkalu's int8 ONNX export (detector-v4-s_int8.onnx).

    The export embeds RT-DETR post-processing: it takes `orig_target_sizes`
    in (width, height) order and returns labels/boxes/scores directly.
    ~3x faster than the fp32 PyTorch path on CPU and needs no torch weights.
    """

    ONNX_FILE = "detector-v4-s_int8.onnx"
    INPUT_SIZE = 640
    ID2LABEL = {0: RegionClass.BUBBLE, 1: RegionClass.TEXT_BUBBLE, 2: RegionClass.TEXT_FREE}

    def __init__(self, model_dir, providers: list[str] | None = None):
        import onnxruntime as ort

        path = str(Path(model_dir) / self.ONNX_FILE)
        self.session = ort.InferenceSession(path, providers=providers or ["CPUExecutionProvider"])

    def detect(self, image: Image.Image, threshold: float = 0.35) -> list[TextRegion]:
        image = image.convert("RGB")
        w, h = image.size
        raw: list[RawBox] = []
        for y0, y1 in compute_tiles(w, h):
            tile = image.crop((0, y0, w, y1))
            raw.extend(self._detect_tile(tile, y0, threshold))
        merged = merge_boxes(raw)
        regions = link_regions(merged)
        log.info("detected %d regions (%d raw boxes)", len(regions), len(raw))
        return regions

    def _detect_tile(self, tile: Image.Image, y_offset: int, threshold: float) -> list[RawBox]:
        import numpy as np

        tw, th = tile.size
        resized = tile.resize((self.INPUT_SIZE, self.INPUT_SIZE))
        x = (np.asarray(resized, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
        sizes = np.array([[tw, th]], dtype=np.int64)  # (width, height) per the export
        labels, boxes, scores = self.session.run(None, {"images": x, "orig_target_sizes": sizes})
        out: list[RawBox] = []
        for label, box, score in zip(labels[0], boxes[0], scores[0]):
            if score < threshold:
                continue
            cls = self.ID2LABEL.get(int(label))
            if cls is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in box)
            x1, y1 = max(0, x1), max(0, y1) + y_offset
            x2, y2 = min(tw, x2), min(th, y2) + y_offset
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(RawBox(bbox=(x1, y1, x2, y2), cls=cls, score=float(score)))
        return out


class BubbleDetector:
    """Fallback PyTorch detector (used only when the ONNX file is absent)."""

    def __init__(self, model_dir, device: str = "cpu"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self._torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModelForObjectDetection.from_pretrained(model_dir).to(device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def detect(self, image: Image.Image, threshold: float = 0.35) -> list[TextRegion]:
        image = image.convert("RGB")
        w, h = image.size
        raw: list[RawBox] = []
        for y0, y1 in compute_tiles(w, h):
            tile = image.crop((0, y0, w, y1))
            raw.extend(self._detect_tile(tile, y0, threshold))
        merged = merge_boxes(raw)
        regions = link_regions(merged)
        log.info("detected %d regions (%d raw boxes)", len(regions), len(raw))
        return regions

    def _detect_tile(self, tile: Image.Image, y_offset: int, threshold: float) -> list[RawBox]:
        tw, th = tile.size
        inputs = self.processor(images=tile, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=[(th, tw)], threshold=threshold
        )[0]
        boxes: list[RawBox] = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = (int(v) for v in box.tolist())
            x1, y1 = max(0, x1), max(0, y1) + y_offset
            x2, y2 = min(tw, x2), min(th, y2) + y_offset
            if x2 <= x1 or y2 <= y1:
                continue
            label_name = self.id2label.get(int(label), "")
            try:
                cls = RegionClass(label_name)
            except ValueError:
                continue
            boxes.append(RawBox(bbox=(x1, y1, x2, y2), cls=cls, score=float(score)))
        return boxes
