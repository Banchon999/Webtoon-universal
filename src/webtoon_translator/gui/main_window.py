"""Main application window."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..core.models import ALL_STAGES, Page, Project, Stage
from ..core.project_io import load_project, save_project
from ..download import manager
from ..paths import data_dir
from ..pipeline.pipeline import PipelineSettings
from ..settings import AppSettings
from .dialogs.settings_dialog import SettingsDialog
from .widgets.canvas import VIEW_CLEANED, VIEW_ORIGINAL, VIEW_TRANSLATED, PageCanvas
from .widgets.glossary_editor import GlossaryEditor
from .widgets.page_list import PageList
from .widgets.progress_dialog import ModelDownloadDialog
from .widgets.region_editor import RegionEditor
from .workers import ExportWorker, PipelineWorker

log = logging.getLogger(__name__)

STAGE_LABELS = {
    "detect": "Detecting",
    "ocr": "Reading text",
    "translate": "Translating",
    "inpaint": "Cleaning",
    "typeset": "Typesetting",
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webtoon Translator")
        self.resize(1400, 900)

        self.app_settings = AppSettings.load()
        self.project = Project(
            source_lang=self.app_settings.source_lang,
            target_lang=self.app_settings.target_lang,
            openrouter_model=self.app_settings.openrouter_model,
        )
        self.project_path: Path | None = None
        self.current_page: Page | None = None
        self._worker: PipelineWorker | None = None
        self._export_worker: ExportWorker | None = None

        self._build_ui()
        self._build_menu()

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # left: pages
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        import_btn = QPushButton("Import images…")
        import_btn.clicked.connect(self._import_dialog)
        self.page_list = PageList()
        self.page_list.pages_dropped.connect(self.add_pages)
        self.page_list.page_selected.connect(self._on_page_selected)
        left_layout.addWidget(import_btn)
        left_layout.addWidget(self.page_list)
        splitter.addWidget(left)

        # center: canvas + view toolbar
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QToolBar()
        self.view_group = QButtonGroup(self)
        for label, mode in (("Original", VIEW_ORIGINAL), ("Cleaned", VIEW_CLEANED), ("Translated", VIEW_TRANSLATED)):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setProperty("view_mode", mode)
            self.view_group.addButton(btn)
            toolbar.addWidget(btn)
            if mode == VIEW_ORIGINAL:
                btn.setChecked(True)
        self.view_group.buttonClicked.connect(
            lambda b: self.canvas.set_mode(b.property("view_mode"))
        )
        fit_btn = QPushButton("Fit width")
        toolbar.addWidget(fit_btn)
        self.canvas = PageCanvas()
        fit_btn.clicked.connect(self.canvas.fit_width)
        self.canvas.region_clicked.connect(self._on_region_clicked)
        center_layout.addWidget(toolbar)
        center_layout.addWidget(self.canvas)
        splitter.addWidget(center)

        # right: actions + region editor
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        self.translate_all_btn = QPushButton("▶ Translate all pages")
        self.translate_page_btn = QPushButton("Translate current page")
        self.retypeset_btn = QPushButton("Re-typeset current page")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        for b in (self.translate_all_btn, self.translate_page_btn, self.retypeset_btn, self.cancel_btn):
            right_layout.addWidget(b)
        self.translate_all_btn.clicked.connect(lambda: self._run_pipeline(all_pages=True))
        self.translate_page_btn.clicked.connect(lambda: self._run_pipeline(all_pages=False))
        self.retypeset_btn.clicked.connect(self._retypeset_current)
        self.cancel_btn.clicked.connect(self._cancel_pipeline)

        self.region_editor = RegionEditor()
        self.region_editor.region_changed.connect(self._on_region_edited)
        self.region_editor.retranslate_requested.connect(self._on_retranslate_region)
        right_layout.addWidget(self.region_editor)
        splitter.addWidget(right)

        splitter.setSizes([220, 880, 300])
        self.setCentralWidget(splitter)

        # status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(260)
        self.progress_bar.setVisible(False)
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.progress_bar)

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")

        def act(menu, text, slot, shortcut=None):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(slot)
            menu.addAction(a)
            return a

        act(file_menu, "Import images…", self._import_dialog, "Ctrl+I")
        file_menu.addSeparator()
        act(file_menu, "Open project…", self._open_project, "Ctrl+O")
        act(file_menu, "Save project", self._save_project, "Ctrl+S")
        act(file_menu, "Save project as…", lambda: self._save_project(save_as=True), "Ctrl+Shift+S")
        file_menu.addSeparator()
        act(file_menu, "Export translated images…", self._export, "Ctrl+E")
        file_menu.addSeparator()
        act(file_menu, "Quit", self.close, "Ctrl+Q")

        tools_menu = self.menuBar().addMenu("&Tools")
        act(tools_menu, "Glossary…", self._edit_glossary, "Ctrl+G")
        act(tools_menu, "Settings…", self._edit_settings, "Ctrl+,")

    # -------------------------------------------------------------- startup --

    def ensure_models(self) -> bool:
        if not manager.missing_models():
            return True
        dialog = ModelDownloadDialog(self)
        if dialog.exec() == ModelDownloadDialog.DialogCode.Accepted:
            return True
        if dialog.error and dialog.error != "cancelled":
            QMessageBox.critical(self, "Download failed", dialog.error)
        return False

    # --------------------------------------------------------------- pages --

    def _import_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import images", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if paths:
            self.add_pages([Path(p) for p in paths])

    def add_pages(self, paths: list[Path]):
        for path in paths:
            page = Page(source_path=path.resolve())
            self.project.pages.append(page)
            self.page_list.add_page(page)
        if self.current_page is None and self.project.pages:
            self.page_list.setCurrentRow(0)

    def _on_page_selected(self, page_id: str):
        self.current_page = self.project.page_by_id(page_id)
        self.canvas.set_page(self.current_page)
        self.canvas.fit_width()
        self.region_editor.set_region(None)

    def _on_region_clicked(self, region_id: str):
        if self.current_page is None:
            return
        region = next((r for r in self.current_page.regions if r.id == region_id), None)
        self.region_editor.set_region(region)
        self.canvas.highlight_region(region_id)

    # ------------------------------------------------------------- pipeline --

    def _pipeline_settings(self) -> PipelineSettings:
        work = data_dir() / "work"
        work.mkdir(parents=True, exist_ok=True)
        return PipelineSettings(
            device_preference=self.app_settings.device_preference,
            detection_threshold=self.app_settings.detection_threshold,
            use_fast_fill=self.app_settings.use_fast_fill,
            rtl_reading_order=self.app_settings.rtl_reading_order,
            api_key=self.app_settings.openrouter_api_key,
            work_dir=work,
            ocr_batch_size=self.app_settings.ocr_batch_size,
            cpu_threads=self.app_settings.cpu_threads,
        )

    def _ensure_worker(self) -> PipelineWorker:
        if self._worker is None or not self._worker.isRunning():
            self._worker = PipelineWorker(self._pipeline_settings())
            self._worker.stage_progress.connect(self._on_stage_progress)
            self._worker.page_done.connect(self._on_page_done)
            self._worker.job_done.connect(self._on_job_done)
            self._worker.failed.connect(self._on_pipeline_failed)
            self._worker.start()
        return self._worker

    def _run_pipeline(self, all_pages: bool, stages: tuple[Stage, ...] = ALL_STAGES):
        if not self.project.pages:
            QMessageBox.information(self, "No pages", "Import images first (File > Import images).")
            return
        if Stage.TRANSLATE in stages and not self.app_settings.openrouter_api_key:
            QMessageBox.warning(
                self,
                "API key required",
                "Set your OpenRouter API key in Tools > Settings before translating.",
            )
            return
        if not self.ensure_models():
            return
        self._sync_project_settings()
        pages = self.project.pages if all_pages else ([self.current_page] if self.current_page else [])
        if not pages:
            return
        self._set_running(True)
        self._ensure_worker().submit(self.project, pages, stages)

    def _retypeset_current(self):
        if self.current_page is None:
            return
        self._set_running(True)
        self._ensure_worker().submit(self.project, [self.current_page], (Stage.TYPESET,))

    def _on_retranslate_region(self, region_id: str):
        if self.current_page is None:
            return
        region = next((r for r in self.current_page.regions if r.id == region_id), None)
        if region is None:
            return
        region.translation_edited = False
        self._set_running(True)
        self._ensure_worker().submit(self.project, [self.current_page], (Stage.TRANSLATE, Stage.TYPESET))

    def _on_region_edited(self, _region_id: str):
        # user edited text/font; rendered output is stale until re-typeset
        self.status_label.setText("Edited — click 'Re-typeset current page' to update the image")

    def _cancel_pipeline(self):
        if self._worker:
            self._worker.cancel()
        self.status_label.setText("Cancelling…")

    def _sync_project_settings(self):
        self.project.source_lang = self.app_settings.source_lang
        self.project.target_lang = self.app_settings.target_lang
        self.project.openrouter_model = self.app_settings.openrouter_model

    def _set_running(self, running: bool):
        for b in (self.translate_all_btn, self.translate_page_btn, self.retypeset_btn):
            b.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.progress_bar.setVisible(running)
        if not running:
            self.progress_bar.setValue(0)

    def _on_stage_progress(self, page_id: str, stage: str, frac: float, msg: str):
        page = self.project.page_by_id(page_id)
        name = Path(page.source_path).name if page else page_id
        label = STAGE_LABELS.get(stage, stage)
        self.status_label.setText(f"{name}: {label} {msg}")
        self.progress_bar.setValue(int(frac * 100))
        self.page_list.set_page_status(page_id, label)

    def _on_page_done(self, page_id: str):
        self.page_list.set_page_status(page_id, "done")
        if self.current_page and self.current_page.id == page_id:
            # show the result
            for btn in self.view_group.buttons():
                if btn.property("view_mode") == VIEW_TRANSLATED:
                    btn.setChecked(True)
            self.canvas.set_mode(VIEW_TRANSLATED)

    def _on_job_done(self):
        self._set_running(False)
        self.status_label.setText("Done")

    def _on_pipeline_failed(self, message: str):
        self._set_running(False)
        self.status_label.setText("Failed")
        QMessageBox.critical(self, "Pipeline error", message)

    # -------------------------------------------------------------- project --

    def _open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "Webtoon project (*.wtproj)")
        if not path:
            return
        try:
            self.project = load_project(Path(path))
        except Exception as e:
            QMessageBox.critical(self, "Open failed", str(e))
            return
        self.project_path = Path(path)
        self.page_list.clear()
        self.current_page = None
        for page in self.project.pages:
            self.page_list.add_page(page)
        if self.project.pages:
            self.page_list.setCurrentRow(0)

    def _save_project(self, save_as: bool = False):
        if self.project_path is None or save_as:
            path, _ = QFileDialog.getSaveFileName(self, "Save project", "untitled.wtproj", "Webtoon project (*.wtproj)")
            if not path:
                return
            self.project_path = Path(path)
        try:
            save_project(self.project, self.project_path)
            self.status_label.setText(f"Saved {self.project_path.name}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _export(self):
        if not self.project.pages:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Export to folder")
        if not out_dir:
            return
        self._export_worker = ExportWorker(
            self.project, Path(out_dir), self.app_settings.export_format, self.app_settings.export_quality
        )
        self._export_worker.done.connect(
            lambda files: self.status_label.setText(f"Exported {len(files)} image(s) to {out_dir}")
        )
        self._export_worker.failed.connect(
            lambda msg: QMessageBox.critical(self, "Export failed", msg)
        )
        self._export_worker.start()

    # --------------------------------------------------------------- dialogs --

    def _edit_glossary(self):
        dialog = GlossaryEditor(self.project.glossary, self)
        if dialog.exec() == GlossaryEditor.DialogCode.Accepted:
            self.project.glossary = dialog.entries()

    def _edit_settings(self):
        dialog = SettingsDialog(self.app_settings, self)
        if dialog.exec() == SettingsDialog.DialogCode.Accepted:
            self.app_settings = dialog.apply()
            self._sync_project_settings()
            # settings affect models/pipeline -> recycle worker lazily
            if self._worker and self._worker.isRunning():
                self._worker.shutdown()
                self._worker = None

    def closeEvent(self, event):
        if self._worker:
            self._worker.shutdown()
            self._worker.wait(3000)
        super().closeEvent(event)
