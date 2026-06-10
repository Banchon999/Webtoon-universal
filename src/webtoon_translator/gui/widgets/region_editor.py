"""Per-region edit panel: OCR text, translation, font controls."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core.models import TextRegion
from ...pipeline.typeset import font_registry


class RegionEditor(QWidget):
    region_changed = Signal(str)  # region_id (text/font edited -> retypeset needed)
    retranslate_requested = Signal(str)  # region_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._region: TextRegion | None = None
        self._building = False

        layout = QVBoxLayout(self)
        self.info_label = QLabel("No region selected")
        layout.addWidget(self.info_label)

        self.enabled_check = QCheckBox("Include this region")
        layout.addWidget(self.enabled_check)

        ocr_group = QGroupBox("Original text (OCR)")
        ocr_layout = QVBoxLayout(ocr_group)
        self.ocr_edit = QPlainTextEdit()
        self.ocr_edit.setMaximumHeight(90)
        ocr_layout.addWidget(self.ocr_edit)
        layout.addWidget(ocr_group)

        tr_group = QGroupBox("Translation")
        tr_layout = QVBoxLayout(tr_group)
        self.translation_edit = QPlainTextEdit()
        self.translation_edit.setMaximumHeight(110)
        tr_layout.addWidget(self.translation_edit)
        self.retranslate_btn = QPushButton("Re-translate this region")
        tr_layout.addWidget(self.retranslate_btn)
        layout.addWidget(tr_group)

        font_group = QGroupBox("Font")
        form = QFormLayout(font_group)
        self.font_combo = QComboBox()
        self.font_combo.addItem("auto")
        for f in font_registry().get("fonts", []):
            self.font_combo.addItem(f["name"])
        self.size_spin = QSpinBox()
        self.size_spin.setRange(0, 200)
        self.size_spin.setSpecialValueText("auto")
        self.stroke_spin = QSpinBox()
        self.stroke_spin.setRange(0, 20)
        form.addRow("Family", self.font_combo)
        form.addRow("Size", self.size_spin)
        form.addRow("Outline", self.stroke_spin)
        layout.addWidget(font_group)
        layout.addStretch(1)

        self.ocr_edit.textChanged.connect(self._on_ocr_edited)
        self.translation_edit.textChanged.connect(self._on_translation_edited)
        self.font_combo.currentTextChanged.connect(self._on_font_changed)
        self.size_spin.valueChanged.connect(self._on_font_changed)
        self.stroke_spin.valueChanged.connect(self._on_font_changed)
        self.enabled_check.toggled.connect(self._on_font_changed)
        self.retranslate_btn.clicked.connect(self._on_retranslate)
        self.setEnabled(False)

    def set_region(self, region: TextRegion | None):
        self._building = True
        self._region = region
        if region is None:
            self.setEnabled(False)
            self.info_label.setText("No region selected")
            self.ocr_edit.clear()
            self.translation_edit.clear()
        else:
            self.setEnabled(True)
            self.info_label.setText(f"{region.cls.value}  ({region.score:.2f})")
            self.enabled_check.setChecked(region.enabled)
            self.ocr_edit.setPlainText(region.ocr_text)
            self.translation_edit.setPlainText(region.translation)
            self.font_combo.setCurrentText(region.font.family)
            self.size_spin.setValue(region.font.size)
            self.stroke_spin.setValue(region.font.stroke_width)
        self._building = False

    def _on_ocr_edited(self):
        if self._building or self._region is None:
            return
        self._region.ocr_text = self.ocr_edit.toPlainText()
        self._region.ocr_edited = True
        self.region_changed.emit(self._region.id)

    def _on_translation_edited(self):
        if self._building or self._region is None:
            return
        self._region.translation = self.translation_edit.toPlainText()
        self._region.translation_edited = True
        self.region_changed.emit(self._region.id)

    def _on_font_changed(self, *_):
        if self._building or self._region is None:
            return
        self._region.enabled = self.enabled_check.isChecked()
        self._region.font.family = self.font_combo.currentText()
        self._region.font.size = self.size_spin.value()
        self._region.font.stroke_width = self.stroke_spin.value()
        self.region_changed.emit(self._region.id)

    def _on_retranslate(self):
        if self._region is not None:
            self.retranslate_requested.emit(self._region.id)
