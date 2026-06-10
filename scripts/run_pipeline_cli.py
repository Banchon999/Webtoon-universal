#!/usr/bin/env python3
"""Headless pipeline runner (thin wrapper around webtoon_translator.cli).

Examples:
    python scripts/run_pipeline_cli.py page1.png page2.png -o out/ --target-lang th
    python scripts/run_pipeline_cli.py strip.png -o out/ --translator dummy   # no API key needed
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webtoon_translator.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
