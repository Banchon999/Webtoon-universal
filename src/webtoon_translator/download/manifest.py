"""Registry of the ML models the app needs, with pinned revisions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    revision: str
    approx_mb: int
    allow_patterns: tuple[str, ...] = field(default_factory=tuple)


MODELS: dict[str, ModelSpec] = {
    "detector": ModelSpec(
        key="detector",
        repo_id="ogkalu/comic-text-and-bubble-detector",
        revision="16e8a622f91fabc6b5b65c96d32d1183f8843546",
        approx_mb=180,
        allow_patterns=("config.json", "preprocessor_config.json", "model.safetensors"),
    ),
    "ocr": ModelSpec(
        key="ocr",
        repo_id="PaddlePaddle/PaddleOCR-VL-1.6",
        revision="66317acc4c9fc17bd154591ce650735cd2855f3e",
        approx_mb=1900,
        # exclude remote-code *.py files: transformers >=5.0 has native paddleocr_vl support
        allow_patterns=(
            "*.json",
            "*.jinja",
            "tokenizer.model",
            "model.safetensors",
        ),
    ),
    "inpaint": ModelSpec(
        key="inpaint",
        repo_id="Carve/LaMa-ONNX",
        revision="c3c0c9e468934d62e79c329e35d82dd09ff8c444",
        approx_mb=210,
        allow_patterns=("lama_fp32.onnx",),
    ),
}
