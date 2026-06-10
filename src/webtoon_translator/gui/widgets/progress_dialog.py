"""First-run model download dialog."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ...download.manifest import MODELS
from ..workers import DownloadWorker


class ModelDownloadDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Downloading AI models")
        self.setModal(True)
        self.resize(420, 160)

        total_mb = sum(s.approx_mb for s in MODELS.values())
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"First run: downloading AI models (~{total_mb} MB total)."))
        self.status_label = QLabel("Starting…")
        layout.addWidget(self.status_label)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        layout.addWidget(self.bar)
        self.cancel_btn = QPushButton("Cancel")
        layout.addWidget(self.cancel_btn)

        self.worker = DownloadWorker(self)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self.accept)
        self.worker.failed.connect(self._on_failed)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self._error: str | None = None

    def exec(self) -> int:
        self.worker.start()
        return super().exec()

    @property
    def error(self) -> str | None:
        return self._error

    def _on_progress(self, model_key: str, filename: str, done: int, total: int):
        self.status_label.setText(f"{model_key}: {filename or 'done'} ({done}/{total})")
        self.bar.setValue(int(100 * done / max(1, total)))

    def _on_failed(self, message: str):
        self._error = message
        self.reject()

    def _on_cancel(self):
        self.worker.cancel()
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Cancelling…")
