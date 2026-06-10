# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Windows build (onedir).

Build from the repo root:
    pyinstaller packaging/webtoon_translator.spec
"""

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

REPO = os.path.abspath(os.path.join(SPECPATH, ".."))

hiddenimports = (
    # transformers Auto* classes resolve model modules dynamically by model_type
    collect_submodules("transformers.models.rt_detr_v2")
    + collect_submodules("transformers.models.paddleocr_vl")
    + collect_submodules("transformers.models.auto")
    + ["pythainlp", "pythainlp.tokenize", "pythainlp.corpus"]
)

datas = (
    collect_data_files("transformers", include_py_files=False)
    + collect_data_files("pythainlp")
    + [
        (os.path.join(REPO, "assets", "fonts"), os.path.join("assets", "fonts")),
        (os.path.join(REPO, "assets", "icons"), os.path.join("assets", "icons")),
    ]
)

excludes = [
    "tkinter",
    "matplotlib",
    "IPython",
    "jupyter",
    "tensorboard",
    "PyQt5",
    "PyQt6",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtQuick",
    "PySide6.QtQml",
    "PySide6.Qt3DCore",
    "PySide6.QtCharts",
    "PySide6.QtMultimedia",
    "PySide6.QtPdf",
]

a = Analysis(
    [os.path.join(REPO, "src", "webtoon_translator", "__main__.py")],
    pathex=[os.path.join(REPO, "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

icon_path = os.path.join(REPO, "assets", "icons", "app.ico")

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WebtoonTranslator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX corrupts torch/Qt DLLs
    console=False,
    icon=icon_path if os.path.exists(icon_path) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="WebtoonTranslator",
)
