"""Application settings dialog."""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
)

from ...pipeline.translator import LANGUAGE_NAMES
from ...settings import AppSettings


class _ModelListWorker(QThread):
    loaded = Signal(list)

    def __init__(self, api_key: str, parent=None):
        super().__init__(parent)
        self.api_key = api_key

    def run(self):
        from ...pipeline.translator import OpenRouterTranslator

        self.loaded.emit(OpenRouterTranslator.list_models(self.api_key))


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(460, 420)
        self.settings = settings

        form = QFormLayout(self)

        self.api_key_edit = QLineEdit(settings.openrouter_api_key)
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("OpenRouter API key", self.api_key_edit)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItem(settings.openrouter_model)
        self.model_combo.setCurrentText(settings.openrouter_model)
        self.refresh_models_btn = QPushButton("Fetch model list")
        self.refresh_models_btn.clicked.connect(self._fetch_models)
        form.addRow("Translation model", self.model_combo)
        form.addRow("", self.refresh_models_btn)

        self.source_combo = QComboBox()
        self.target_combo = QComboBox()
        for code, name in LANGUAGE_NAMES.items():
            self.source_combo.addItem(f"{name} ({code})", code)
            if code != "auto":
                self.target_combo.addItem(f"{name} ({code})", code)
        self._set_combo(self.source_combo, settings.source_lang)
        self._set_combo(self.target_combo, settings.target_lang)
        form.addRow("Source language", self.source_combo)
        form.addRow("Target language", self.target_combo)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.05, 0.95)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(settings.detection_threshold)
        form.addRow("Detection threshold", self.threshold_spin)

        self.fast_fill_check = QCheckBox("Fast-fill plain bubbles (skip AI inpainting when uniform)")
        self.fast_fill_check.setChecked(settings.use_fast_fill)
        form.addRow("", self.fast_fill_check)

        self.rtl_check = QCheckBox("Right-to-left reading order (manga)")
        self.rtl_check.setChecked(settings.rtl_reading_order)
        form.addRow("", self.rtl_check)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu"])
        self.device_combo.setCurrentText(settings.device_preference)
        form.addRow("Compute device", self.device_combo)

        self.ocr_batch_spin = QSpinBox()
        self.ocr_batch_spin.setRange(1, 16)
        self.ocr_batch_spin.setValue(settings.ocr_batch_size)
        self.ocr_batch_spin.setToolTip("Bubbles recognized per OCR pass. Higher = faster but uses more RAM.")
        form.addRow("OCR batch size", self.ocr_batch_spin)

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 64)
        self.threads_spin.setSpecialValueText("all cores")
        self.threads_spin.setValue(settings.cpu_threads)
        form.addRow("CPU threads", self.threads_spin)

        self.format_combo = QComboBox()
        self.format_combo.addItems(["png", "jpg", "webp"])
        self.format_combo.setCurrentText(settings.export_format)
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(50, 100)
        self.quality_spin.setValue(settings.export_quality)
        form.addRow("Export format", self.format_combo)
        form.addRow("Export quality", self.quality_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self._worker: _ModelListWorker | None = None

    @staticmethod
    def _set_combo(combo: QComboBox, code: str):
        idx = combo.findData(code)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _fetch_models(self):
        self.refresh_models_btn.setEnabled(False)
        self.refresh_models_btn.setText("Fetching…")
        self._worker = _ModelListWorker(self.api_key_edit.text().strip(), self)
        self._worker.loaded.connect(self._on_models_loaded)
        self._worker.start()

    def _on_models_loaded(self, models: list):
        current = self.model_combo.currentText()
        if models:
            self.model_combo.clear()
            self.model_combo.addItems(models)
            idx = self.model_combo.findText(current, Qt.MatchFlag.MatchExactly)
            self.model_combo.setCurrentIndex(idx) if idx >= 0 else self.model_combo.setCurrentText(current)
            self.refresh_models_btn.setText("Fetch model list")
        else:
            self.refresh_models_btn.setText("Fetch failed (offline?)")
        self.refresh_models_btn.setEnabled(True)

    def apply(self) -> AppSettings:
        s = self.settings
        s.openrouter_api_key = self.api_key_edit.text().strip()
        s.openrouter_model = self.model_combo.currentText().strip()
        s.source_lang = self.source_combo.currentData()
        s.target_lang = self.target_combo.currentData()
        s.detection_threshold = self.threshold_spin.value()
        s.use_fast_fill = self.fast_fill_check.isChecked()
        s.rtl_reading_order = self.rtl_check.isChecked()
        s.device_preference = self.device_combo.currentText()
        s.export_format = self.format_combo.currentText()
        s.export_quality = self.quality_spin.value()
        s.ocr_batch_size = self.ocr_batch_spin.value()
        s.cpu_threads = self.threads_spin.value()
        s.save()
        return s
