from webtoon_translator.download import manager
from webtoon_translator.download.manifest import MODELS, ModelSpec


def test_signature_changes_with_patterns():
    a = ModelSpec(key="x", repo_id="r", revision="abc", approx_mb=1, allow_patterns=("m.onnx",))
    b = ModelSpec(key="x", repo_id="r", revision="abc", approx_mb=1, allow_patterns=("m.safetensors",))
    assert a.signature != b.signature


def test_is_downloaded_invalidated_by_manifest_change(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "models_dir", lambda: tmp_path)
    spec = MODELS["detector"]
    d = tmp_path / "detector"
    d.mkdir()
    (d / ".complete").write_text("old-revision:model.safetensors")
    assert not manager.is_downloaded("detector")
    (d / ".complete").write_text(spec.signature)
    assert manager.is_downloaded("detector")
