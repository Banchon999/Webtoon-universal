"""Path resolution helpers: app data dirs, frozen-app awareness, bundled assets."""

from __future__ import annotations

import sys
from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "WebtoonTranslator"


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def bundle_root() -> Path:
    """Root directory containing bundled assets (source tree or PyInstaller _MEIPASS)."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    # src/webtoon_translator/paths.py -> repo root
    return Path(__file__).resolve().parents[2]


def assets_dir() -> Path:
    return bundle_root() / "assets"


def fonts_dir() -> Path:
    return assets_dir() / "fonts"


def data_dir() -> Path:
    d = Path(user_data_dir(APP_NAME, appauthor=False))
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    d = data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d
