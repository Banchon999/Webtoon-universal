"""Application entry point."""

from __future__ import annotations

import logging
import sys


def self_test() -> int:
    """Import-level smoke test for the frozen build (run with --self-test)."""
    try:
        import onnxruntime  # noqa: F401
        import PySide6  # noqa: F401
        import torch  # noqa: F401
        from transformers.models.paddleocr_vl import modeling_paddleocr_vl  # noqa: F401
        from transformers.models.rt_detr_v2 import modeling_rt_detr_v2  # noqa: F401

        from .pipeline import detector, inpaint, ocr, translator, typeset  # noqa: F401
        from .pipeline.typeset import pick_font_file

        pick_font_file("auto", "th")
        pick_font_file("auto", "en")
        print("self-test OK")
        return 0
    except Exception as e:
        print(f"self-test FAILED: {e}", file=sys.stderr)
        return 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if "--self-test" in sys.argv:
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
