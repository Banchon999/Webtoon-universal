"""Glossary editor dialog: editable table + CSV/JSON import/export."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ...core.glossary import load_glossary, save_glossary
from ...core.models import GlossaryEntry


class GlossaryEditor(QDialog):
    def __init__(self, entries: list[GlossaryEntry], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Glossary")
        self.resize(560, 480)

        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Source term", "Translation", "Note", "On"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setColumnWidth(0, 160)
        self.table.setColumnWidth(1, 160)
        self.table.setColumnWidth(2, 140)
        self.table.setColumnWidth(3, 36)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add")
        del_btn = QPushButton("Remove")
        import_btn = QPushButton("Import…")
        export_btn = QPushButton("Export…")
        for b in (add_btn, del_btn, import_btn, export_btn):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        add_btn.clicked.connect(lambda: self._add_row())
        del_btn.clicked.connect(self._remove_selected)
        import_btn.clicked.connect(self._import)
        export_btn.clicked.connect(self._export)

        for e in entries:
            self._add_row(e)

    def _add_row(self, entry: GlossaryEntry | None = None):
        row = self.table.rowCount()
        self.table.insertRow(row)
        e = entry or GlossaryEntry(source="", target="")
        self.table.setItem(row, 0, QTableWidgetItem(e.source))
        self.table.setItem(row, 1, QTableWidgetItem(e.target))
        self.table.setItem(row, 2, QTableWidgetItem(e.note))
        check = QTableWidgetItem()
        check.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        check.setCheckState(Qt.CheckState.Checked if e.enabled else Qt.CheckState.Unchecked)
        self.table.setItem(row, 3, check)

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def entries(self) -> list[GlossaryEntry]:
        result = []
        for row in range(self.table.rowCount()):
            source = (self.table.item(row, 0) or QTableWidgetItem()).text().strip()
            if not source:
                continue
            result.append(
                GlossaryEntry(
                    source=source,
                    target=(self.table.item(row, 1) or QTableWidgetItem()).text().strip(),
                    note=(self.table.item(row, 2) or QTableWidgetItem()).text().strip(),
                    enabled=self.table.item(row, 3).checkState() == Qt.CheckState.Checked,
                )
            )
        return result

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import glossary", "", "Glossary (*.csv *.json)")
        if not path:
            return
        try:
            for e in load_glossary(Path(path)):
                self._add_row(e)
        except Exception as ex:
            QMessageBox.warning(self, "Import failed", str(ex))

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export glossary", "glossary.csv", "Glossary (*.csv *.json)")
        if not path:
            return
        try:
            save_glossary(Path(path), self.entries())
        except Exception as ex:
            QMessageBox.warning(self, "Export failed", str(ex))
