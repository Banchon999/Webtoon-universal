"""GUI application entry point."""

from __future__ import annotations

import io
import logging
import sys


def _guard_windowed_stdio() -> None:
    """In windowed (no-console) builds sys.stdout/stderr are None; libraries
    like tqdm/transformers write to them during import and inference, so give
    them a sink instead of letting attribute errors (or blocked writes) happen.
    """
    if sys.stdout is None:
        sys.stdout = io.StringIO()
    if sys.stderr is None:
        sys.stderr = io.StringIO()
    sys.__stdout__ = sys.stdout
    sys.__stderr__ = sys.stderr


def main() -> int:
    _guard_windowed_stdio()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if "--self-test" in sys.argv:
        from .cli import self_test

        return self_test()

    from PySide6.QtWidgets import QApplication

    from .gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Webtoon Translator")
    app.setOrganizationName("WebtoonTranslator")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
