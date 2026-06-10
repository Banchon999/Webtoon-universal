"""Text recognition with PaddlePaddle/PaddleOCR-VL-1.6 via transformers.

The model is a ~1B-param vision-language model; each detected text region is
cropped and recognized with the "OCR:" task prompt.
"""

from __future__ import annotations

import logging

from PIL import Image

from ..core.models import TextRegion

log = logging.getLogger(__name__)

CROP_PAD = 6
MAX_NEW_TOKENS = 256


class OcrEngine:
    def __init__(self, model_dir, device: str = "cpu"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.device = device
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = (
            AutoModelForImageTextToText.from_pretrained(model_dir, dtype=dtype).to(device).eval()
        )
        self.processor = AutoProcessor.from_pretrained(model_dir)

    def recognize(self, crop: Image.Image) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop.convert("RGB")},
                    {"type": "text", "text": "OCR:"},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        text = self.processor.decode(new_tokens, skip_special_tokens=True)
        return clean_ocr_text(text)

    def recognize_region(self, page_image: Image.Image, region: TextRegion) -> str:
        x1, y1, x2, y2 = region.bbox
        w, h = page_image.size
        crop = page_image.crop(
            (max(0, x1 - CROP_PAD), max(0, y1 - CROP_PAD), min(w, x2 + CROP_PAD), min(h, y2 + CROP_PAD))
        )
        return self.recognize(crop)


def clean_ocr_text(text: str) -> str:
    """Collapse whitespace/newlines; bubble text is re-wrapped at typesetting time."""
    return " ".join(text.split()).strip()
