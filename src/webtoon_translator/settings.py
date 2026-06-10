"""Typed application settings persisted with QSettings.

Kept import-light: QSettings is only imported when the GUI runs, so the
pipeline/CLI never needs Qt.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class AppSettings:
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemini-2.5-flash"
    source_lang: str = "auto"
    target_lang: str = "th"
    detection_threshold: float = 0.35
    use_fast_fill: bool = True
    rtl_reading_order: bool = False
    device_preference: str = "auto"  # auto | cpu
    export_format: str = "png"  # png | jpg | webp
    export_quality: int = 92
    default_font: str = "auto"

    @classmethod
    def load(cls) -> AppSettings:
        from PySide6.QtCore import QSettings

        qs = QSettings("WebtoonTranslator", "WebtoonTranslator")
        kwargs = {}
        for f in fields(cls):
            if not qs.contains(f.name):
                continue
            raw = qs.value(f.name)
            if f.type in ("float",):
                kwargs[f.name] = float(raw)
            elif f.type in ("int",):
                kwargs[f.name] = int(raw)
            elif f.type in ("bool",):
                kwargs[f.name] = raw in (True, "true", "True", "1", 1)
            else:
                kwargs[f.name] = str(raw)
        return cls(**kwargs)

    def save(self) -> None:
        from PySide6.QtCore import QSettings

        qs = QSettings("WebtoonTranslator", "WebtoonTranslator")
        for f in fields(self):
            qs.setValue(f.name, getattr(self, f.name))
        qs.sync()
