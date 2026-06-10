"""Page list with thumbnails and drag-drop import."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QListWidget, QListWidgetItem

from ...core.models import Page

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
THUMB_SIZE = QSize(96, 128)

STAGE_ICONS = {0: "⬜", 1: "🟨", 2: "✅"}


class PageList(QListWidget):
    pages_dropped = Signal(list)  # list[Path]
    page_selected = Signal(str)  # page_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setIconSize(THUMB_SIZE)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.DropOnly)
        self.currentItemChanged.connect(self._on_current_changed)

    def _on_current_changed(self, current, _previous):
        if current is not None:
            self.page_selected.emit(current.data(Qt.ItemDataRole.UserRole))

    def add_page(self, page: Page):
        pixmap = QPixmap(str(page.source_path))
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        item = QListWidgetItem(pixmap, Path(page.source_path).name)
        item.setData(Qt.ItemDataRole.UserRole, page.id)
        self.addItem(item)

    def set_page_status(self, page_id: str, text: str):
        for i in range(self.count()):
            item = self.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == page_id:
                base = item.text().split(" — ")[0]
                item.setText(f"{base} — {text}" if text else base)
                return

    # -- drag & drop ---------------------------------------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths: list[Path] = []
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_dir():
                paths.extend(sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS))
            elif p.suffix.lower() in IMAGE_EXTS:
                paths.append(p)
        if paths:
            self.pages_dropped.emit(paths)
        event.acceptProposedAction()
