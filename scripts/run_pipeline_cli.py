#!/usr/bin/env python3
"""Headless pipeline runner for testing and batch use.

Examples:
    python scripts/run_pipeline_cli.py page1.png page2.png -o out/ --target-lang th
    python scripts/run_pipeline_cli.py strip.png -o out/ --translator dummy   # no API key needed
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webtoon_translator.core.glossary import load_glossary  # noqa: E402
from webtoon_translator.core.models import Page, Project, Stage  # noqa: E402
from webtoon_translator.download import manager  # noqa: E402
from webtoon_translator.pipeline.export import export_page  # noqa: E402
from webtoon_translator.pipeline.pipeline import PipelineSettings, TranslationPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate webtoon/comic images headlessly")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("output"))
    parser.add_argument("--source-lang", default="auto")
    parser.add_argument("--target-lang", default="th")
    parser.add_argument("--api-key", default="", help="OpenRouter API key (or env OPENROUTER_API_KEY)")
    parser.add_argument("--model", default="google/gemini-2.5-flash", help="OpenRouter model id")
    parser.add_argument("--translator", choices=["openrouter", "dummy"], default="openrouter")
    parser.add_argument("--glossary", type=Path, help="glossary CSV/JSON file")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--no-fast-fill", action="store_true", help="always use LaMa inpainting")
    parser.add_argument("--skip-translate", action="store_true")
    parser.add_argument("--format", choices=["png", "jpg", "webp"], default="png")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import os

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if args.translator == "openrouter" and not args.skip_translate and not api_key:
        parser.error("--api-key / OPENROUTER_API_KEY required (or use --translator dummy)")

    print("Checking models...")
    missing = manager.missing_models()
    if missing:
        total = sum(s.approx_mb for s in missing)
        print(f"Downloading {len(missing)} model(s), ~{total} MB ...")
        manager.ensure_all(
            progress_cb=lambda key, f, done, n: print(f"  [{key}] {done}/{n} {f}")
        )

    project = Project(source_lang=args.source_lang, target_lang=args.target_lang, openrouter_model=args.model)
    if args.glossary:
        project.glossary = load_glossary(args.glossary)
    project.pages = [Page(source_path=p.resolve()) for p in args.images]

    work_dir = Path(tempfile.mkdtemp(prefix="wt_pipeline_"))
    settings = PipelineSettings(
        detection_threshold=args.threshold,
        use_fast_fill=not args.no_fast_fill,
        api_key=api_key,
        use_dummy_translator=args.translator == "dummy",
        work_dir=work_dir,
    )

    def progress(page_id: str, stage: Stage, frac: float, msg: str) -> None:
        if msg:
            print(f"  [{stage.value}] {frac * 100:3.0f}% {msg}")

    pipeline = TranslationPipeline(settings, progress_cb=progress)
    stages = tuple(s for s in Stage if not (args.skip_translate and s == Stage.TRANSLATE))

    for page in project.pages:
        print(f"Processing {page.source_path.name} ...")
        pipeline.run(project, [page], stages=stages)
        out = export_page(page, args.output, args.format)
        print(f"  -> {out}")
        for r in page.text_regions():
            print(f"     [{r.cls.value}] {r.ocr_text!r} -> {r.translation!r}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
