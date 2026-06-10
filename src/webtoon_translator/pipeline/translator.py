"""Machine translation via OpenRouter (OpenAI-compatible chat completions).

All bubbles of a page are sent in one request, numbered and in reading order,
so the model sees cross-bubble context. Glossary entries whose source term
appears on the page are injected into the system prompt as hard constraints.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

import httpx

from ..core.glossary import relevant_entries
from ..core.models import GlossaryEntry

log = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
APP_REFERER = "https://github.com/Banchon999/Webtoon-universal"
APP_TITLE = "Webtoon Translator"

LANGUAGE_NAMES = {
    "auto": "the source language (detect it)",
    "en": "English",
    "th": "Thai",
    "ko": "Korean",
    "ja": "Japanese",
    "zh": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "id": "Indonesian",
    "vi": "Vietnamese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "ms": "Malay",
    "tl": "Filipino",
    "my": "Burmese",
    "km": "Khmer",
    "lo": "Lao",
}


class TranslationError(RuntimeError):
    pass


@dataclass
class TranslatorConfig:
    api_key: str
    model: str = "google/gemini-2.5-flash"
    source_lang: str = "auto"
    target_lang: str = "th"
    timeout: float = 90.0
    max_retries: int = 3


def build_system_prompt(cfg: TranslatorConfig, glossary: list[GlossaryEntry], texts: list[str]) -> str:
    src = LANGUAGE_NAMES.get(cfg.source_lang, cfg.source_lang)
    tgt = LANGUAGE_NAMES.get(cfg.target_lang, cfg.target_lang)
    lines = [
        "You are an expert comic/webtoon localizer.",
        f"Translate each numbered line from {src} into {tgt}.",
        "Keep the tone natural and conversational, matching comic dialogue.",
        "Preserve interjections and onomatopoeia feel; adapt idioms naturally.",
        "Do NOT add explanations. Keep translations concise enough to fit speech bubbles.",
    ]
    used = relevant_entries(glossary, texts)
    if used:
        lines.append("Glossary (always use these exact translations):")
        for e in used:
            note = f"  # {e.note}" if e.note else ""
            lines.append(f'- "{e.source}" -> "{e.target}"{note}')
    lines.append(
        'Respond with ONLY a JSON object: {"translations": ["...", "...", ...]} '
        "with exactly one string per input line, same order."
    )
    return "\n".join(lines)


def build_user_prompt(texts: list[str]) -> str:
    return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))


def parse_translations(content: str, expected: int) -> list[str] | None:
    """Parse the model reply defensively. Returns None when count mismatches."""
    content = content.strip()
    # strip markdown fences
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", content, re.DOTALL)
    if fence:
        content = fence.group(1)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if isinstance(data, dict):
        items = data.get("translations")
    elif isinstance(data, list):
        items = data
    else:
        return None
    if not isinstance(items, list):
        return None
    items = [str(x) for x in items]
    return items if len(items) == expected else None


class OpenRouterTranslator:
    def __init__(self, cfg: TranslatorConfig, glossary: list[GlossaryEntry] | None = None):
        self.cfg = cfg
        self.glossary = glossary or []
        self._client = httpx.Client(
            base_url=OPENROUTER_BASE,
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "HTTP-Referer": APP_REFERER,
                "X-Title": APP_TITLE,
            },
            timeout=cfg.timeout,
        )

    def close(self) -> None:
        self._client.close()

    def _chat(self, messages: list[dict]) -> str:
        payload = {"model": self.cfg.model, "messages": messages}
        last_error: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self._client.post("/chat/completions", json=payload)
                if resp.status_code == 429:
                    delay = float(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                    time.sleep(min(delay, 30))
                    continue
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    raise TranslationError(str(data["error"]))
                return data["choices"][0]["message"]["content"]
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last_error = e
                if attempt < self.cfg.max_retries:
                    time.sleep(2**attempt)
        raise TranslationError(f"OpenRouter request failed: {last_error}")

    def translate_texts(self, texts: list[str]) -> list[str]:
        """Translate a page worth of texts; falls back to per-line requests."""
        if not texts:
            return []
        system = build_system_prompt(self.cfg, self.glossary, texts)
        content = self._chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": build_user_prompt(texts)},
            ]
        )
        parsed = parse_translations(content, len(texts))
        if parsed is not None:
            return parsed
        log.warning("batch parse failed (%d lines); falling back to per-line", len(texts))
        return [self._translate_single(t) for t in texts]

    def _translate_single(self, text: str) -> str:
        system = build_system_prompt(self.cfg, self.glossary, [text])
        content = self._chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": build_user_prompt([text])},
            ]
        )
        parsed = parse_translations(content, 1)
        return parsed[0] if parsed else content.strip()

    @staticmethod
    def list_models(api_key: str = "") -> list[str]:
        """Model ids from OpenRouter for the settings combo. Empty list offline."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            resp = httpx.get(f"{OPENROUTER_BASE}/models", headers=headers, timeout=15)
            resp.raise_for_status()
            return sorted(m["id"] for m in resp.json().get("data", []))
        except Exception as e:
            log.warning("could not fetch OpenRouter model list: %s", e)
            return []


class DummyTranslator:
    """Offline stand-in for tests/CI: upper-cases text and tags it."""

    def __init__(self, *args, **kwargs):
        pass

    def translate_texts(self, texts: list[str]) -> list[str]:
        return [f"[TH] {t}" for t in texts]

    def close(self) -> None:
        pass
