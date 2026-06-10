"""Page canvas: zoomable image view with region overlays."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from ...core.models import Page, RegionClass

CLASS_COLORS = {
    RegionClass.BUBBLE: QColor(80, 160, 255),
    RegionClass.TEXT_BUBBLE: QColor(80, 220, 120),
    RegionClass.TEXT_FREE: QColor(255, 170, 60),
}

VIEW_ORIGINAL = "original"
VIEW_CLEANED = "cleaned"
VIEW_TRANSLATED = "translated"


class RegionItem(QGraphicsRectItem):
    def __init__(self, region_id: str, rect: QRectF, color: QColor):
        super().__init__(rect)
        self.region_id = region_id
        self._color = color
        self.setPen(QPen(color, 3))
        self.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 28)))
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable)

    def set_highlight(self, on: bool):
        pen = QPen(self._color, 5 if on else 3)
        if on:
            pen.setStyle(Qt.PenStyle.DashLine)
        self.setPen(pen)


class PageCanvas(QGraphicsView):
    region_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._region_items: dict[str, RegionItem] = {}
        self._page: Page | None = None
        self._mode = VIEW_ORIGINAL
        self._show_overlays = True

    # -- content -----------------------------------------------------------

    def set_page(self, page: Page | None):
        self._page = page
        self.refresh()

    def set_mode(self, mode: str):
        self._mode = mode
        self.refresh()

    def set_overlays_visible(self, visible: bool):
        self._show_overlays = visible
        for item in self._region_items.values():
            item.setVisible(visible)

    def current_image_path(self):
        if self._page is None:
            return None
        if self._mode == VIEW_TRANSLATED and self._page.rendered_image_path:
            return self._page.rendered_image_path
        if self._mode in (VIEW_CLEANED, VIEW_TRANSLATED) and self._page.cleaned_image_path:
            return self._page.cleaned_image_path
        return self._page.source_path

    def refresh(self):
        self._scene.clear()
        self._region_items.clear()
        self._pixmap_item = None
        if self._page is None:
            return
        path = self.current_image_path()
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        if self._mode != VIEW_TRANSLATED:
            for region in self._page.regions:
                x1, y1, x2, y2 = region.bbox
                item = RegionItem(
                    region.id, QRectF(x1, y1, x2 - x1, y2 - y1), CLASS_COLORS[region.cls]
                )
                item.setVisible(self._show_overlays)
                self._scene.addItem(item)
                self._region_items[region.id] = item

    def highlight_region(self, region_id: str | None):
        for rid, item in self._region_items.items():
            item.set_highlight(rid == region_id)
        if region_id and region_id in self._region_items:
            self.ensureVisible(self._region_items[region_id], 50, 50)

    def fit_width(self):
        if self._pixmap_item:
            rect = self._pixmap_item.boundingRect()
            self.fitInView(
                QRectF(rect.x(), rect.y(), rect.width(), min(rect.height(), rect.width() * 1.2)),
                Qt.AspectRatioMode.KeepAspectRatio,
            )

    # -- interaction ---------------------------------------------------------

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if isinstance(item, RegionItem):
            self.region_clicked.emit(item.region_id)
        super().mousePressEvent(event)
