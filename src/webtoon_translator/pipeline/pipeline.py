"""Pipeline orchestrator: detect -> ocr -> translate -> inpaint -> typeset.

GUI-free by design: progress is reported through plain callbacks and
cancellation through a threading.Event, so the same code runs under the Qt
worker thread, the CLI, and tests.

Models are loaded lazily on first use and stay resident for subsequent runs.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..core.device import onnx_providers, pick_device
from ..core.models import ALL_STAGES, Page, Project, Stage
from ..download import manager
from .detector import BubbleDetector, reading_order
from .inpaint import LamaInpainter
from .ocr import OcrEngine
from .translator import DummyTranslator, OpenRouterTranslator, TranslatorConfig
from .typeset import render_page

log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None  # webtoon strips legitimately exceed PIL's default limit

# (page_id, stage, fraction_done_within_stage, message)
ProgressCb = Callable[[str, Stage, float, str], None]


class PipelineCancelled(Exception):
    pass


@dataclass
class PipelineSettings:
    device_preference: str = "auto"
    detection_threshold: float = 0.35
    use_fast_fill: bool = True
    rtl_reading_order: bool = False
    api_key: str = ""
    use_dummy_translator: bool = False  # tests/CI
    work_dir: Path = field(default_factory=lambda: Path("."))


class TranslationPipeline:
    def __init__(
        self,
        settings: PipelineSettings,
        progress_cb: ProgressCb | None = None,
        cancel_event: threading.Event | None = None,
    ):
        self.settings = settings
        self.progress_cb = progress_cb or (lambda *a: None)
        self.cancel_event = cancel_event or threading.Event()
        self._detector: BubbleDetector | None = None
        self._ocr: OcrEngine | None = None
        self._inpainter: LamaInpainter | None = None

    # -- lazy model accessors -------------------------------------------------

    @property
    def detector(self) -> BubbleDetector:
        if self._detector is None:
            self._detector = BubbleDetector(
                manager.model_path("detector"), pick_device(self.settings.device_preference)
            )
        return self._detector

    @property
    def ocr(self) -> OcrEngine:
        if self._ocr is None:
            self._ocr = OcrEngine(
                manager.model_path("ocr"), pick_device(self.settings.device_preference)
            )
        return self._ocr

    @property
    def inpainter(self) -> LamaInpainter:
        if self._inpainter is None:
            self._inpainter = LamaInpainter(manager.model_path("inpaint"), onnx_providers())
        return self._inpainter

    def _make_translator(self, project: Project):
        if self.settings.use_dummy_translator:
            return DummyTranslator()
        cfg = TranslatorConfig(
            api_key=self.settings.api_key,
            model=project.openrouter_model,
            source_lang=project.source_lang,
            target_lang=project.target_lang,
        )
        return OpenRouterTranslator(cfg, project.glossary)

    # -- helpers ---------------------------------------------------------------

    def _check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise PipelineCancelled()

    def _progress(self, page: Page, stage: Stage, frac: float, msg: str = "") -> None:
        self.progress_cb(page.id, stage, frac, msg)

    def _cache_path(self, page: Page, kind: str) -> Path:
        d = self.settings.work_dir / kind
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{page.id}.png"

    # -- stages ----------------------------------------------------------------

    def run_detect(self, page: Page) -> None:
        self._check_cancel()
        self._progress(page, Stage.DETECT, 0.0, "detecting bubbles")
        image = Image.open(page.source_path)
        page.size = image.size
        regions = self.detector.detect(image, self.settings.detection_threshold)
        page.regions = reading_order(regions, self.settings.rtl_reading_order)
        page.invalidate(Stage.DETECT)
        page.mark_done(Stage.DETECT)
        self._progress(page, Stage.DETECT, 1.0, f"{len(page.text_regions())} text regions")

    def run_ocr(self, page: Page) -> None:
        image = Image.open(page.source_path).convert("RGB")
        targets = [r for r in page.text_regions() if not r.ocr_edited]
        for i, region in enumerate(targets):
            self._check_cancel()
            self._progress(page, Stage.OCR, i / max(1, len(targets)), f"OCR {i + 1}/{len(targets)}")
            region.ocr_text = self.ocr.recognize_region(image, region)
        page.mark_done(Stage.OCR)
        self._progress(page, Stage.OCR, 1.0)

    def run_translate(self, page: Page, project: Project, translator=None) -> None:
        self._check_cancel()
        self._progress(page, Stage.TRANSLATE, 0.0, "translating")
        own_translator = translator is None
        if own_translator:
            translator = self._make_translator(project)
        try:
            targets = [r for r in page.text_regions() if r.ocr_text and not r.translation_edited]
            if targets:
                translations = translator.translate_texts([r.ocr_text for r in targets])
                for region, tr in zip(targets, translations):
                    region.translation = tr
            page.mark_done(Stage.TRANSLATE)
            self._progress(page, Stage.TRANSLATE, 1.0)
        finally:
            if own_translator:
                translator.close()

    def run_inpaint(self, page: Page) -> None:
        self._check_cancel()
        self._progress(page, Stage.INPAINT, 0.0, "removing original text")
        image = cv2.cvtColor(np.array(Image.open(page.source_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        cleaned = self.inpainter.clean_page(image, page.regions, self.settings.use_fast_fill)
        out = self._cache_path(page, "cleaned")
        cv2.imwrite(str(out), cleaned)
        page.cleaned_image_path = out
        page.mark_done(Stage.INPAINT)
        self._progress(page, Stage.INPAINT, 1.0)

    def run_typeset(self, page: Page, project: Project) -> None:
        self._check_cancel()
        self._progress(page, Stage.TYPESET, 0.0, "typesetting")
        base_path = page.cleaned_image_path or page.source_path
        base = Image.open(base_path)
        rendered = render_page(base, page.regions, project.target_lang)
        out = self._cache_path(page, "rendered")
        rendered.save(out)
        page.rendered_image_path = out
        page.mark_done(Stage.TYPESET)
        self._progress(page, Stage.TYPESET, 1.0)

    # -- entry point -------------------------------------------------------------

    def run(
        self,
        project: Project,
        pages: list[Page] | None = None,
        stages: tuple[Stage, ...] = ALL_STAGES,
    ) -> None:
        pages = pages if pages is not None else project.pages
        translator = None
        try:
            if Stage.TRANSLATE in stages:
                translator = self._make_translator(project)
            for page in pages:
                if Stage.DETECT in stages and Stage.DETECT not in page.stages_done:
                    self.run_detect(page)
                if Stage.OCR in stages:
                    self.run_ocr(page)
                if Stage.TRANSLATE in stages:
                    self.run_translate(page, project, translator)
                if Stage.INPAINT in stages:
                    self.run_inpaint(page)
                if Stage.TYPESET in stages:
                    self.run_typeset(page, project)
        finally:
            if translator is not None:
                translator.close()
