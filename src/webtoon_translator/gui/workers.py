"""Background workers. All model work happens on these threads; the GUI
receives plain-data signals only."""

from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from ..core.models import Page, Project, Stage
from ..download import manager
from ..pipeline.pipeline import (
    PipelineCancelled,
    PipelineSettings,
    TranslationPipeline,
)

log = logging.getLogger(__name__)


class DownloadWorker(QThread):
    progress = Signal(str, str, int, int)  # model_key, filename, files_done, files_total
    failed = Signal(str)
    done = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        try:
            manager.ensure_all(
                progress_cb=lambda *a: self.progress.emit(*a),
                cancel_check=self._cancel.is_set,
            )
            self.done.emit()
        except InterruptedError:
            self.failed.emit("cancelled")
        except Exception as e:
            log.error("model download failed:\n%s", traceback.format_exc())
            self.failed.emit(str(e))


class PipelineWorker(QThread):
    """Long-lived worker owning the models. Jobs are queued; models stay
    resident between runs so re-runs after edits are fast."""

    stage_progress = Signal(str, str, float, str)  # page_id, stage, frac, message
    page_done = Signal(str)
    job_done = Signal()
    failed = Signal(str)

    def __init__(self, settings: PipelineSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._jobs: list[tuple[Project, list[Page], tuple[Stage, ...]]] = []
        self._job_available = threading.Event()
        self._cancel = threading.Event()
        self._quit = False
        self._lock = threading.Lock()

    def submit(self, project: Project, pages: list[Page], stages: tuple[Stage, ...]):
        with self._lock:
            self._jobs.append((project, pages, stages))
        self._cancel.clear()
        self._job_available.set()

    def cancel(self):
        self._cancel.set()

    def shutdown(self):
        self._quit = True
        self._cancel.set()
        self._job_available.set()

    def run(self):
        pipeline = TranslationPipeline(
            self.settings,
            progress_cb=lambda pid, stage, frac, msg: self.stage_progress.emit(
                pid, stage.value, frac, msg
            ),
            cancel_event=self._cancel,
        )
        while not self._quit:
            self._job_available.wait()
            if self._quit:
                break
            with self._lock:
                if not self._jobs:
                    self._job_available.clear()
                    continue
                project, pages, stages = self._jobs.pop(0)
            try:
                for page in pages:
                    pipeline.run(project, [page], stages)
                    self.page_done.emit(page.id)
                self.job_done.emit()
            except PipelineCancelled:
                self.job_done.emit()
            except Exception as e:
                log.error("pipeline failed:\n%s", traceback.format_exc())
                self.failed.emit(str(e))


class ExportWorker(QThread):
    progress = Signal(int, int)
    done = Signal(list)
    failed = Signal(str)

    def __init__(self, project: Project, out_dir: Path, fmt: str, quality: int, parent=None):
        super().__init__(parent)
        self.project = project
        self.out_dir = out_dir
        self.fmt = fmt
        self.quality = quality

    def run(self):
        from ..pipeline.export import export_page

        try:
            results = []
            n = len(self.project.pages)
            for i, page in enumerate(self.project.pages):
                self.progress.emit(i, n)
                p = export_page(page, self.out_dir, self.fmt, self.quality)
                if p:
                    results.append(str(p))
            self.done.emit(results)
        except Exception as e:
            log.error("export failed:\n%s", traceback.format_exc())
            self.failed.emit(str(e))
