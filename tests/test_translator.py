import json

import httpx
import respx

from webtoon_translator.core.models import GlossaryEntry
from webtoon_translator.pipeline.translator import (
    OPENROUTER_BASE,
    DummyTranslator,
    OpenRouterTranslator,
    TranslatorConfig,
    build_system_prompt,
    parse_translations,
)


def _cfg():
    return TranslatorConfig(api_key="sk-test", model="test/model", target_lang="th", max_retries=1)


def test_parse_translations_plain_json():
    content = json.dumps({"translations": ["สวัสดี", "ลาก่อน"]})
    assert parse_translations(content, 2) == ["สวัสดี", "ลาก่อน"]


def test_parse_translations_fenced():
    content = '```json\n{"translations": ["A"]}\n```'
    assert parse_translations(content, 1) == ["A"]


def test_parse_translations_bare_list():
    assert parse_translations('["x", "y"]', 2) == ["x", "y"]


def test_parse_translations_count_mismatch():
    assert parse_translations('{"translations": ["only one"]}', 2) is None


def test_parse_translations_garbage():
    assert parse_translations("no json here", 3) is None


def test_glossary_only_relevant_terms_in_prompt():
    glossary = [
        GlossaryEntry(source="용사", target="ผู้กล้า"),
        GlossaryEntry(source="마왕", target="จอมมาร"),
        GlossaryEntry(source="안녕", target="สวัสดี", enabled=False),
    ]
    prompt = build_system_prompt(_cfg(), glossary, ["용사가 나타났다", "안녕하세요"])
    assert "ผู้กล้า" in prompt
    assert "จอมมาร" not in prompt  # not on this page
    assert "สวัสดี" not in prompt  # disabled entry


@respx.mock
def test_translate_texts_batch():
    route = respx.post(f"{OPENROUTER_BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"translations": ["หนึ่ง", "สอง"]}'}}]},
        )
    )
    tr = OpenRouterTranslator(_cfg())
    assert tr.translate_texts(["one", "two"]) == ["หนึ่ง", "สอง"]
    assert route.called
    req = route.calls[0].request
    assert req.headers["authorization"] == "Bearer sk-test"
    tr.close()


@respx.mock
def test_translate_falls_back_per_line_on_bad_batch():
    responses = iter(
        [
            httpx.Response(200, json={"choices": [{"message": {"content": "not json at all"}}]}),
            httpx.Response(200, json={"choices": [{"message": {"content": '{"translations": ["A1"]}'}}]}),
            httpx.Response(200, json={"choices": [{"message": {"content": '{"translations": ["B1"]}'}}]}),
        ]
    )
    respx.post(f"{OPENROUTER_BASE}/chat/completions").mock(side_effect=lambda req: next(responses))
    tr = OpenRouterTranslator(_cfg())
    assert tr.translate_texts(["a", "b"]) == ["A1", "B1"]
    tr.close()


@respx.mock
def test_retry_on_transport_error():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"translations": ["ok"]}'}}]})

    respx.post(f"{OPENROUTER_BASE}/chat/completions").mock(side_effect=handler)
    tr = OpenRouterTranslator(_cfg())
    assert tr.translate_texts(["x"]) == ["ok"]
    assert calls["n"] == 2
    tr.close()


def test_dummy_translator():
    assert DummyTranslator().translate_texts(["hi"]) == ["[TH] hi"]
