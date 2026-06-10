"""Text removal (cleaning) with Carve/LaMa-ONNX.

The ONNX export takes fixed 512x512 inputs:
    image: float32 [1,3,512,512], RGB, scaled to 0-1
    mask:  float32 [1,1,512,512], binary (1 = inpaint here)
Output [3,512,512] is already in the 0-255 range.

We never resize the page: text regions are grouped into 512x512 windows and
each window is inpainted and composited back only where the mask is set.
A fast path fills near-uniform bubble interiors directly without the model.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..core.models import RegionClass, TextRegion

log = logging.getLogger(__name__)

LAMA_SIZE = 512
MASK_DILATE_PX = 7
UNIFORM_STD_THRESHOLD = 8.0


def build_mask(shape_hw: tuple[int, int], regions: list[TextRegion]) -> np.ndarray:
    """Full-page uint8 mask (255 inside dilated text boxes)."""
    mask = np.zeros(shape_hw, dtype=np.uint8)
    for r in regions:
        if not r.enabled or r.cls == RegionClass.BUBBLE:
            continue
        x1, y1, x2, y2 = r.bbox
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    if MASK_DILATE_PX > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * MASK_DILATE_PX + 1, 2 * MASK_DILATE_PX + 1)
        )
        mask = cv2.dilate(mask, kernel)
    return mask


def mask_windows(mask: np.ndarray, win: int = LAMA_SIZE) -> list[tuple[int, int, int, int]]:
    """Group mask components into (x, y, w, h) windows of size <= win x win.

    Components larger than the window are covered by a grid of overlapping
    windows.
    """
    h, w = mask.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = [
        (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        for i in range(1, n)
    ]
    windows: list[tuple[int, int, int, int]] = []
    for bx, by, bw, bh in boxes:
        step = win - 64  # overlap between grid windows for oversized components
        for gy in range(by, by + bh, step):
            for gx in range(bx, bx + bw, step):
                wx = max(0, min(w - win, gx - max(0, (win - bw) // 2)))
                wy = max(0, min(h - win, gy - max(0, (win - bh) // 2)))
                if w <= win:
                    wx = 0
                if h <= win:
                    wy = 0
                windows.append((wx, wy, min(win, w), min(win, h)))
                if gx + win >= bx + bw:
                    break
            if gy + win >= by + bh:
                break
    # dedupe overlapping identical windows
    return sorted(set(windows))


def is_uniform_region(image: np.ndarray, region: TextRegion) -> tuple[bool, np.ndarray]:
    """Check whether the border ring around the region's text box is near-uniform.

    Returns (uniform?, median BGR color of the ring).
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = region.bbox
    pad = MASK_DILATE_PX + 6
    ox1, oy1 = max(0, x1 - pad), max(0, y1 - pad)
    ox2, oy2 = min(w, x2 + pad), min(h, y2 + pad)
    outer = image[oy1:oy2, ox1:ox2]
    ring = np.ones(outer.shape[:2], dtype=bool)
    ring[(y1 - oy1) : (y2 - oy1), (x1 - ox1) : (x2 - ox1)] = False
    pixels = outer[ring]
    if pixels.size == 0:
        return False, np.zeros(3)
    std = pixels.reshape(-1, 3).std(axis=0).max()
    median = np.median(pixels.reshape(-1, 3), axis=0)
    return std < UNIFORM_STD_THRESHOLD, median


class LamaInpainter:
    ONNX_FILE = "lama_fp32.onnx"

    def __init__(self, model_dir, providers: list[str] | None = None):
        import onnxruntime as ort

        path = str(model_dir / self.ONNX_FILE) if hasattr(model_dir, "__fspath__") else model_dir
        self.session = ort.InferenceSession(path, providers=providers or ["CPUExecutionProvider"])

    def _inpaint_window(self, window_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint a window (any size <= 512, padded internally) and return same-size BGR."""
        wh, ww = window_bgr.shape[:2]
        pad_b, pad_r = LAMA_SIZE - wh, LAMA_SIZE - ww
        img = cv2.copyMakeBorder(window_bgr, 0, pad_b, 0, pad_r, cv2.BORDER_REPLICATE)
        msk = cv2.copyMakeBorder(mask, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT, value=0)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_in = rgb.transpose(2, 0, 1)[None]
        mask_in = ((msk > 0).astype(np.float32))[None, None]
        out = self.session.run(None, {"image": image_in, "mask": mask_in})[0][0]
        out = np.clip(out, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)[:wh, :ww]

    def clean_page(
        self,
        image_bgr: np.ndarray,
        regions: list[TextRegion],
        use_fast_fill: bool = True,
    ) -> np.ndarray:
        result = image_bgr.copy()
        model_regions: list[TextRegion] = []
        for r in regions:
            if not r.enabled or r.cls == RegionClass.BUBBLE:
                continue
            if use_fast_fill:
                uniform, color = is_uniform_region(image_bgr, r)
                if uniform:
                    m = build_mask(image_bgr.shape[:2], [r])
                    result[m > 0] = color.astype(np.uint8)
                    continue
            model_regions.append(r)

        if not model_regions:
            return result

        mask = build_mask(image_bgr.shape[:2], model_regions)
        for wx, wy, ww, wh in mask_windows(mask):
            win_img = result[wy : wy + wh, wx : wx + ww]
            win_mask = mask[wy : wy + wh, wx : wx + ww]
            if win_mask.max() == 0:
                continue
            inpainted = self._inpaint_window(win_img, win_mask)
            sel = win_mask > 0
            win_img[sel] = inpainted[sel]
        return result
